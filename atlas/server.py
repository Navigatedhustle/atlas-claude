"""Atlas web server: one page on :8100, a WebSocket for answers + audio, and an ASR proxy."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from collections import deque

import httpx
import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from atlas import brain
from atlas.config import ASSISTANT_NAME, BRIEF_QR_URL, FEATURES, LLM_URL as LLM_URL_, OUT_DIR, PERSONAS, ROOT, SPEECH_URL
from atlas.speech import Speech
from atlas.sysmon import SysMon

app = FastAPI(title=ASSISTANT_NAME)
WEB = ROOT / "web"
OUT_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=WEB), name="static")
app.mount("/out", StaticFiles(directory=OUT_DIR), name="out")

llm = brain.LLM()
s1 = brain.SystemOne(llm)
speech = Speech()
mon = SysMon()
STATE = {"last": None, "latency": deque(maxlen=50), "vision_ms": None, "vision_ok": False, "warm": False,
         "kev_ms": None, "llm_route_ms": None, "llm_first_token_ms": None}

GOLDEN = {
    "enterprise": [
        "What's our projected Q4 revenue and profit?",
        "Which five deals are most at risk this quarter, and why?",
        "Should we build a new sales center in Phoenix, Denver, or Albuquerque?",
        "What happens to Q4 profit if our Commit stage win rate drops ten points?",
        "Which region is furthest behind plan, and by how much?",
        "How many reps do we need to hire to hit next year's target?",
        "How does our pipeline coverage compare to this time last year?",
        "Give me a thirty second board summary.",
        "What did the last QBR say about risk in our weakest region?",
        "Summarize the board's concerns about margin.",
        "Which accounts does our pricing policy flag for discount approval?",
    ],
    "university": [
        "Based on our grant pipeline, what's projected research funding for FY27?",
        "Which sponsors are we most dependent on, and what's the risk?",
        "Where should we open our next research center?",
        "What's our projected fall enrollment yield?",
        "Which PIs have the strongest NIH track record?",
        "What does the sponsor guidance say about cost sharing?",
        "Summarize our enrollment strategy risks.",
    ],
    "developer": [
        "Write a SQL query for the top ten accounts by open pipeline, and run it.",
        "Write a Python script that exports the Q4 forecast by region to CSV, then run it.",
        "Explain how the Q4 forecast is calculated and show me the code.",
        "Write unit tests for the site scoring function and run them.",
        "Find open deals whose close date has already passed, and show me the query.",
        "Build a REST endpoint that returns pipeline coverage by region, and test it.",
        "Plot win rate by stage from the historical deals.",
        "How fast is the model running right now, and how do I connect my own tools to it?",
    ],
}


# Second-brain golden questions (hidden from the drawer when ATLAS_DOCS=0).
BRAIN_GOLDEN = set(GOLDEN["enterprise"][8:]) | set(GOLDEN["university"][4:])


@app.on_event("startup")
async def startup():
    asyncio.create_task(_warm())


async def _warm():
    """Warm every model once so no question pays a cold start."""
    for _ in range(60):
        await asyncio.gather(llm.health(), s1.health(), speech.health())
        if llm.ok and speech.ok:
            break
        await asyncio.sleep(2)
    try:
        async for _ in brain.answer("What's our projected Q4 revenue and profit?", "enterprise", llm):
            pass
        brain.STATS["questions"] -= 1
        brain.STATS["grounding_pass"] = max(brain.STATS["grounding_pass"] - 1, 0)
        from atlas.analytics import university as U, enterprise as E
        U.grant_forecast(); E.deal_risk(5)
        if speech.ok:
            await speech.tts("Atlas is ready.")
        if await s1.health():
            await s1.decide(E.deal_risk(5)["decisions"][:1])
    except Exception as e:  # noqa: BLE001
        print("warmup error", e)
    STATE["warm"] = True


async def _rewarm():
    try:
        await asyncio.sleep(1.0)
        async for _ in brain.answer("Give me one sentence about our quarter.", "enterprise", llm):
            pass
        brain.STATS["questions"] -= 1
        if speech.ok:
            await speech.tts("Ready.")
    except Exception as e:  # noqa: BLE001
        print("rewarm error", e)


@app.get("/")
async def index():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/config")
async def config():
    golden = {p: [q for q in qs if FEATURES["docs"] or q not in BRAIN_GOLDEN]
              for p, qs in GOLDEN.items() if p != "developer" or FEATURES["developer"]}
    return {"name": ASSISTANT_NAME, "personas": PERSONAS, "golden": golden, "qr_url": BRIEF_QR_URL, "features": FEATURES}


@app.get("/api/stats")
async def stats():
    async def vision_health():
        try:
            async with httpx.AsyncClient(timeout=0.5) as c:
                STATE["vision_ok"] = (await c.get("http://127.0.0.1:8130/health")).json().get("ok", False)
        except Exception:
            STATE["vision_ok"] = False
    was = (llm.ok, speech.ok)
    await asyncio.gather(llm.health(), s1.health(), speech.health(), vision_health())
    if STATE["warm"] and ((llm.ok and not was[0]) or (speech.ok and not was[1])):
        asyncio.create_task(_rewarm())  # a service came back after an outage: warm it before the next visitor
    net = mon.network()
    mem = mon.memory(ROOT / "run")
    STATE["peak_gb"] = max(STATE.get("peak_gb", 0.0), mem["atlas_gb"])
    mem["atlas_peak_gb"] = STATE["peak_gb"]
    gpu = mon.gpu()
    lat = list(STATE["latency"])
    last = brain.STATS["last"].get("timings", {})
    models = [
        {"id": "llm", "name": "NVIDIA Nemotron 3.5 Lightning 30B-A3B", "role": "Brain", "ok": llm.ok,
         "ms": STATE["llm_route_ms"], "detail": "Q4 GGUF · llama.cpp · thinking off"},
        {"id": "asr", "name": "Nemotron Speech 0.6B", "role": "Ears", "ok": speech.ok,
         "ms": speech.last_asr_ms, "detail": "NeMo-Speech.cpp · CUDA"},
        {"id": "tts", "name": "Magpie TTS 357M", "role": "Voice", "ok": speech.ok,
         "ms": speech.last_tts_ms, "detail": "Q8_0 · CUDA graphs"},
        {"id": "kev", "name": "System One decisions 4B", "role": "Decisions", "ok": s1.ok, "ms": STATE["kev_ms"],
         "detail": "calibrated typed decisions"},
        {"id": "vision", "name": "RF-DETR Nano", "role": "Eyes", "ok": STATE["vision_ok"], "ms": STATE["vision_ms"],
         "detail": "person detection"},
    ]
    return {"net": net, "mem": mem, "gpu": gpu, "models": models, "warm": STATE["warm"],
            "tokens": brain.STATS["llm_tokens"], "questions": brain.STATS["questions"],
            "grounding": {"pass": brain.STATS["grounding_pass"], "fail": brain.STATS["grounding_fail"]},
            "last_timings": last, "latency": lat[-10:]}


@app.post("/api/dev/run")
async def dev_run(req: Request):
    """The RUN IN SANDBOX button: execute exactly the code on screen and return what really happened."""
    if not FEATURES["developer"]:
        return JSONResponse({"error": "developer mode is switched off"}, status_code=404)
    from atlas import sandbox
    d = await req.json()
    lang = "sql" if d.get("lang") == "sql" else "python"
    entry = ["-m", "pytest", "-q", "-p", "no:cacheprovider", "main.py"] if d.get("tests") else None
    return await asyncio.to_thread(sandbox.run, lang, d.get("code", "")[:20000], None, entry)


@app.get("/api/dev/endpoint")
async def dev_endpoint(req: Request):
    """Live facts for the CONNECT YOUR OWN TOOLS card (no synthetic load is generated by polling this)."""
    host = req.headers.get("host", "127.0.0.1:8100")
    out = {"base_url": f"http://{host}/v1", "model": "nemotron-3.5-lightning", "lan": host.split(":")[0] not in ("127.0.0.1", "localhost"),
           "bind": os.environ.get("ATLAS_HOST", "127.0.0.1")}
    try:
        async with httpx.AsyncClient(base_url=LLM_URL_, timeout=1.5) as c:
            props, slots = (await c.get("/props")).json(), (await c.get("/slots")).json()
        out.update({"n_ctx": props.get("default_generation_settings", {}).get("n_ctx"), "slots": len(slots),
                    "busy": sum(1 for x in slots if x.get("is_processing")), "ok": True,
                    "model": props.get("model_alias") or out["model"]})
    except Exception:
        out["ok"] = False
    lg = brain.STATS.get("last_gen")
    if lg:
        out["last_tok_s"], out["last_n"], out["last_age_s"] = lg["tok_s"], lg["n"], time.time() - lg["t"]
    return out


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def llm_proxy(path: str, req: Request):
    """Standard chat-completions API for your own tools, proxied to the model server on loopback."""
    if path not in ("models", "chat/completions", "completions", "embeddings"):
        return JSONResponse({"error": "not found"}, status_code=404)
    body = await req.body()
    client = httpx.AsyncClient(base_url=LLM_URL_, timeout=httpx.Timeout(300.0, connect=2.0))
    up = client.build_request(req.method, f"/v1/{path}", content=body, headers={"content-type": req.headers.get("content-type", "application/json")})
    r = await client.send(up, stream=True)

    async def gen():
        try:
            async for chunk in r.aiter_raw():
                yield chunk
        finally:
            await r.aclose()
            await client.aclose()
    from fastapi.responses import StreamingResponse
    return StreamingResponse(gen(), status_code=r.status_code, media_type=r.headers.get("content-type"))


@app.get("/api/brain/graph")
async def brain_graph(persona: str = "enterprise"):
    from atlas.knowledge import graph as G
    if not FEATURES["brain"] or persona not in ("enterprise", "university"):
        return JSONResponse({"error": "brain off"}, status_code=404)
    g = await asyncio.to_thread(G.load, persona)
    return Response(content=json.dumps(g, separators=(",", ":")), media_type="application/json")


@app.get("/api/brain/node/{node_id:path}")
async def brain_node(node_id: str):
    from atlas.knowledge import graph as G
    return await asyncio.to_thread(G.record, node_id)


@app.get("/api/brain/knows")
async def brain_knows(persona: str = "enterprise"):
    """HOW ATLAS KNOWS cards. Every number is computed now from the data, the graph, the index and the counters."""
    from atlas.knowledge import graph as G
    g = await asyncio.to_thread(G.load, persona)
    out = {"persona": persona, "data": await asyncio.to_thread(G.data_connected, persona), "graph": G.stats(g),
           "net": mon.network(), "now": time.time()}
    try:
        from atlas.knowledge import docs
        out["docs"] = await asyncio.to_thread(docs.stats, persona)
    except Exception as e:  # noqa: BLE001
        out["docs"] = {"error": str(e)[:120]}
    out["decision"] = decision_model_status()
    return out


@app.post("/api/brain/feed")
async def brain_feed(persona: str = "enterprise"):
    if not (FEATURES["feed"] and FEATURES["docs"]):
        return JSONResponse({"ok": False, "error": "Feed the brain is switched off"}, status_code=404)
    from atlas.knowledge import docs
    try:
        return await asyncio.to_thread(docs.feed, persona, LLM_URL_)
    except httpx.ConnectError:
        return {"ok": False, "error": "The local language model is not running, so no document was written. Nothing was ingested."}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"Feed failed before ingest: {str(e)[:160]}. Nothing was ingested."}


@app.post("/api/brain/feed/reset")
async def brain_feed_reset(persona: str = "enterprise"):
    from atlas.knowledge import docs
    return await asyncio.to_thread(docs.reset_feed, persona)


def decision_model_status() -> dict:
    p = ROOT / "eval" / "finetune" / "finetune.json"  # measured result of the optional fine-tune (committed)
    ft = json.loads(p.read_text()) if p.exists() else None
    return {"base": "Kev-4B System One (released checkpoint jaredpalmer/kev-4b)", "ok": s1.ok,
            "calibration": "as released: typed yes/no and choice probabilities",
            "finetune": ft, "finetune_text": (ft or {}).get("ui_text") or "Not fine-tuned: general calibration"}


@app.post("/api/net/reset")
async def net_reset():
    mon.reset()
    return {"ok": True}


@app.post("/api/latency")
async def report_latency(req: Request):
    """Browser reports measured end-to-end latency (release of the button -> first audio out)."""
    d = await req.json()
    d["t"] = time.time()
    STATE["latency"].append(d)
    return {"ok": True}


@app.post("/api/brief")
async def make_brief():
    if not STATE["last"]:
        return JSONResponse({"error": "Ask a question first."}, status_code=400)
    from atlas.brief import build_brief
    paths = await asyncio.to_thread(build_brief, STATE["last"], BRIEF_QR_URL)
    return paths


@app.post("/api/say")
async def say(req: Request):
    d = await req.json()
    try:
        wav, _ = await speech.tts(d["text"][:300])
        return Response(content=wav, media_type="audio/wav")
    except Exception:
        return JSONResponse({"error": "tts unavailable"}, status_code=503)


@app.post("/api/vision")
async def vision(req: Request):
    body = await req.body()
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            t = time.perf_counter()
            r = await c.post("http://127.0.0.1:8130/detect", content=body, headers={"content-type": "image/jpeg"})
            STATE["vision_ms"] = (time.perf_counter() - t) * 1000
            STATE["vision_ok"] = True
            return r.json()
    except Exception:
        STATE["vision_ok"] = False
        return {"ok": False, "people": [], "error": "vision service not running"}


@app.websocket("/ws")
async def ws_main(ws: WebSocket):
    await ws.accept()
    task: asyncio.Task | None = None
    convo = {"previous": None, "persona": None}

    async def run(msg):
        aid = msg.get("id")
        _send = ws.send_json

        async def send(d):
            await _send({**d, "id": aid})
        q = msg["text"].strip()
        persona = msg.get("persona", "enterprise")
        voice = msg.get("voice", True)
        lang = "es" if msg.get("lang") == "es" else "en"
        tts_tasks = []
        t0 = time.perf_counter()
        await send({"type": "question", "text": q, "persona": persona, "source": msg.get("source")})
        pending_decisions: list = []
        first_audio_ready = asyncio.Event() if (voice and speech.ok) else None

        async def synth(i, text):
            try:
                wav, ms = await speech.tts(text, PERSONAS[persona]["voice"], "es-ES" if lang == "es" else "en-US")
                await send({"type": "audio", "seq": i, "wav": base64.b64encode(wav).decode(), "tts_ms": ms,
                                    "server_ms": (time.perf_counter() - t0) * 1000})
                # System One runs after the first audio chunk so it never competes with time-to-first-audio on the GPU.
                if i == 0:
                    if first_audio_ready is not None:
                        first_audio_ready.set()
                    if pending_decisions:
                        asyncio.create_task(decide(pending_decisions.pop(), send))
            except Exception as e:  # noqa: BLE001
                if i == 0 and first_audio_ready is not None:
                    first_audio_ready.set()
                await send({"type": "audio_error", "seq": i, "error": str(e)[:100]})

        previous = convo["previous"] if convo["persona"] == persona else None
        if not brain.is_follow_up(q):
            convo["previous"], convo["persona"] = q, persona
        try:
            await _answer_loop(q, persona, lang, voice, first_audio_ready, previous, send, synth, decide, tts_tasks,
                               pending_decisions, t0)
        except asyncio.CancelledError:
            for t in tts_tasks:
                t.cancel()
            raise

    async def _answer_loop(q, persona, lang, voice, first_audio_ready, previous, send, synth, decide, tts_tasks,
                           pending_decisions, t0):
        seq = 0
        result = None
        if persona == "developer" and FEATURES["developer"]:
            events = _dev_events(q, lang, first_audio_ready, send)
        else:
            events = brain.answer(q, persona, llm, lang=lang, first_audio_ready=first_audio_ready, previous=previous)
        async for ev in events:
            if ev["type"] in ("code", "run", "endpoint"):
                await send(ev)
            elif ev["type"] == "route":
                await send(ev)
            elif ev["type"] == "result":
                result = ev["result"]
                await send({"type": "result", "result": result, "ms": ev.get("ms")})
                if FEATURES["brain"] and persona in ("enterprise", "university"):
                    asyncio.create_task(_provenance(persona, result, send))
                if not result.get("decisions"):
                    await send({"type": "decisions", "engine": "none", "items": [], "ms": 0})
                elif not (voice and speech.ok):
                    asyncio.create_task(decide(result["decisions"], send))
                else:
                    pending_decisions.append(result["decisions"])
            elif ev["type"] == "sentence":
                await send({"type": "sentence", "seq": seq, "text": ev["text"], "source": ev["source"]})
                if voice and speech.ok:
                    tts_tasks.append(asyncio.create_task(synth(seq, ev["text"])))
                seq += 1
            elif ev["type"] == "done":
                STATE["llm_route_ms"] = ev["timings"].get("llm_route_ms", STATE["llm_route_ms"])
                STATE["last"] = {"question": q, "persona": persona, "result": result, "answer": ev["text"],
                                 "grounding": ev["grounding"], "timings": ev["timings"]}
                if tts_tasks:
                    await asyncio.gather(*tts_tasks)
                if pending_decisions:
                    asyncio.create_task(decide(pending_decisions.pop(), send))
                await send({**ev, "sentences": seq, "audio": bool(voice and tts_tasks)})

    async def _provenance(persona, result, send):
        from atlas.knowledge import graph as G
        try:
            pr = await asyncio.to_thread(G.provenance, persona, result.get("tool"), result)
            await send({"type": "provenance", **pr})
        except Exception as e:  # noqa: BLE001
            print("provenance error", e)
            await send({"type": "provenance", "nodes": [], "rows": 0, "tables": [], "tool": result.get("tool"), "error": str(e)[:100]})

    async def _dev_events(q, lang, first_audio_ready, send):
        from atlas import dev
        brain.STATS["questions"] += 1
        t0 = time.perf_counter()
        prefix, result, tool, picked, timings = [], None, None, None, {}
        async for ev in dev.answer(q, llm, send, ws.headers.get("host", "127.0.0.1:8100")):
            if ev["type"] == "spoken_prefix":
                prefix = ev["sentences"]
                continue
            if ev["type"] == "route":
                tool, picked = ev["tool"], ev["by"]
                timings["llm_route_ms"] = ev.get("ms")
            if ev["type"] == "result":
                result = ev["result"]
                timings["tool_ms"] = ev.get("ms")
            yield ev
        async for ev in brain.narrate_result(q, "developer", result, llm, True, lang, first_audio_ready, None, timings,
                                             t0, tool, picked, already_spoken=prefix):
            yield ev

    async def decide(decisions, send=None):
        send = send or ws.send_json
        try:
            d = await s1.decide(decisions)
            STATE["kev_ms"] = d["ms"]
            await send({"type": "decisions", **d})
        except Exception as e:  # noqa: BLE001
            await send({"type": "decisions", "engine": "unavailable", "items": [], "error": str(e)[:100]})

    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "ask":
                if task and not task.done():
                    task.cancel()
                task = asyncio.create_task(run(msg))
            elif msg.get("type") == "stop":
                if task and not task.done():
                    task.cancel()
                await ws.send_json({"type": "stopped"})
    except WebSocketDisconnect:
        if task:
            task.cancel()


@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    """Browser streams 16 kHz PCM16 while the button is held; we relay to the local streaming ASR."""
    await ws.accept()
    url = SPEECH_URL.replace("http", "ws") + "/v1/audio/transcriptions/realtime"
    try:
        async with websockets.connect(url, max_size=None) as up:
            await up.send(json.dumps({"type": "session.update", "session": {"sample_rate": 16000, "automatic_punctuation": True}}))
            t_commit = None

            async def downstream():
                nonlocal t_commit
                async for m in up:
                    if isinstance(m, bytes):
                        continue
                    d = json.loads(m)
                    typ = d.get("type", "")
                    if typ.endswith("transcription.delta"):
                        await ws.send_json({"type": "partial", "text": d.get("delta") or d.get("text") or ""})
                    elif typ.endswith("transcription.completed"):
                        ms = (time.perf_counter() - t_commit) * 1000 if t_commit else None
                        speech.last_asr_ms = ms
                        await ws.send_json({"type": "final", "text": d.get("transcript") or d.get("text") or "", "asr_ms": ms})
                    elif typ == "error":
                        await ws.send_json({"type": "error", "error": d})

            down = asyncio.create_task(downstream())
            while True:
                m = await ws.receive()
                if m.get("bytes"):
                    await up.send(m["bytes"])
                elif m.get("text"):
                    d = json.loads(m["text"])
                    if d.get("type") == "commit":
                        t_commit = time.perf_counter()
                        await up.send(json.dumps({"type": "input_audio_buffer.commit"}))
                elif m.get("type") == "websocket.disconnect":
                    break
            down.cancel()
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        try:
            await ws.send_json({"type": "error", "error": str(e)[:200]})
        except Exception:
            pass
