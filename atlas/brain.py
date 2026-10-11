"""The brain: the LLM picks a tool and narrates; Python does every number; grounding gates every sentence."""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import AsyncIterator

import httpx

from atlas import grounding
from atlas.analytics import TOOLS, run_tool, tool_schemas
from atlas.config import ASSISTANT_NAME, KEV_URL, LLM_URL, PERSONAS
from atlas.router import route

STATS = {"llm_tokens": 0, "llm_calls": 0, "questions": 0, "grounding_pass": 0, "grounding_fail": 0,
         "last": {}}

ROUTE_SYS = ("You route an executive's question to exactly one analytics tool for {org}. "
             "Always call one tool. Never answer directly.")

NARRATE_SYS = ("You are {name}, a voice assistant for the {role} of {org}. Answer the question in 2 or 3 spoken "
               "sentences, 30 to 55 words: lead with the answer, then the most important supporting facts. Use only numbers that appear in FACTS, copied exactly "
               "as written there, including $ and % signs. Never calculate, round, approximate or invent numbers "
               "(no 'over 40', 'about half', 'nearly'). If the FACTS do not answer the question, say that plainly and give the "
               "closest relevant fact instead; never stretch a fact to fit. Never answer general knowledge or do arithmetic. "
               "No lists, no markdown, no greetings.")

NARRATE_ES = (" Answer in Spanish (neutral Latin American). Keep every number exactly as written in FACTS, in English "
              "format, e.g. $644.7M or 13.3% (they are converted for speech later).")

SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9$])")


class LLM:
    def __init__(self):
        self.client = httpx.AsyncClient(base_url=LLM_URL, timeout=httpx.Timeout(20.0, connect=1.0))
        self.ok = False

    async def health(self) -> bool:
        try:
            r = await self.client.get("/health", timeout=1.0)
            self.ok = r.status_code == 200
        except Exception:
            self.ok = False
        return self.ok

    async def pick_tool(self, question: str, persona: str, context: str | None = None) -> tuple[str | None, dict, dict]:
        """The LLM picks the tool. Output is grammar-constrained to a valid tool name (3-6 tokens), which keeps
        routing near 200 ms; literal arguments (cities, stage, points) are then extracted deterministically."""
        org = PERSONAS[persona]["org"]
        names = list(TOOLS[persona])
        menu = "\n".join(f"{n}: {TOOLS[persona][n][1]}" for n in names)
        grammar = "root ::= " + " | ".join(f'"{n}"' for n in names)
        body = {"messages": [{"role": "system", "content": ROUTE_SYS.format(org=org) + "\nTools:\n" + menu +
                              "\nReply with the tool name only."},
                             {"role": "user", "content": (f"(Earlier question: {context})\n" if context else "") + question}],
                "grammar": grammar, "temperature": 0, "max_tokens": 12}
        t = time.perf_counter()
        r = await self.client.post("/v1/chat/completions", json=body, timeout=6.0)
        r.raise_for_status()
        d = r.json()
        meta = {"ms": (time.perf_counter() - t) * 1000, "tokens": d.get("usage", {}).get("total_tokens", 0)}
        STATS["llm_tokens"] += meta["tokens"]
        STATS["llm_calls"] += 1
        name = (d["choices"][0]["message"].get("content") or "").strip()
        if name not in TOOLS[persona]:
            return None, {}, meta
        from atlas.router import extract_args
        return name, extract_args(name, question), meta

    def narrate_messages(self, question: str, persona: str, result: dict, lang: str = "en",
                         context: str | None = None) -> list[dict]:
        p = PERSONAS[persona]
        facts = "\n".join(f"- {f}" for f in result["facts"] + [f"{i['label']}: {i['text']}" for i in result.get("inputs", [])])
        sys_prompt = NARRATE_SYS.format(name=ASSISTANT_NAME, role=p["role"], org=p["org"]) + (NARRATE_ES if lang == "es" else "")
        if persona == "developer":
            sys_prompt += (" The code has already been written and shown on screen; FACTS describe what really happened when it "
                           "ran. Never write, quote or read out code, SQL, column lists or file contents. Say in plain words "
                           "what ran and what it found. If it failed, say so and give the error in plain words.")
        ctx = f"EARLIER QUESTION: {context}\n" if context else ""
        tail = "\nRespond in Spanish. Copy numbers exactly as written in FACTS." if lang == "es" else ""
        return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"{ctx}QUESTION: {question}\nFACTS:\n{facts}{tail}"}]

    async def template(self, messages: list[dict]) -> str:
        r = await self.client.post("/apply-template", json={"messages": messages}, timeout=3.0)
        r.raise_for_status()
        return r.json()["prompt"]

    async def complete_stream(self, prompt: str, n_predict: int, temperature: float = 0.2) -> AsyncIterator[str]:
        body = {"prompt": prompt, "n_predict": n_predict, "temperature": temperature, "stream": True, "cache_prompt": True}
        async with self.client.stream("POST", "/completion", json=body) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    d = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                tok = d.get("content")
                if tok:
                    STATS["llm_tokens"] += 1
                    yield tok
                if d.get("stop"):
                    tm = d.get("timings") or {}
                    STATS["llm_tokens"] += tm.get("prompt_n", 0)
                    if tm.get("predicted_n", 0) >= 8:
                        STATS["last_gen"] = {"tok_s": tm.get("predicted_per_second"), "n": tm.get("predicted_n"), "t": time.time()}
                    break
        STATS["llm_calls"] += 1

    async def narrate_paused(self, messages: list[dict], ctl: dict, prefix: str = "") -> AsyncIterator[str]:
        """Stream the narration, but stop generating once ctl['pause'] is set (first chunk ready) and resume as a
        cached continuation after ctl['resume'] fires, so the first chunk's TTS gets the GPU to itself.
        `prefix` pre-fills the start of the answer (used to pin the output language)."""
        prompt = await self.template(messages) + prefix
        raw = ""
        if prefix:
            yield prefix
        budget = 120
        async for tok in self.complete_stream(prompt, budget):
            raw += tok
            yield tok
            if ctl.get("pause"):
                break
        else:
            return
        ev = ctl.get("resume")
        if ev is not None:
            try:
                await asyncio.wait_for(ev.wait(), 1.5)
            except asyncio.TimeoutError:
                pass
        n_used = max(len(raw) // 4, 1)
        async for tok in self.complete_stream(prompt + raw, max(budget - n_used, 40)):  # prompt already ends with prefix
            yield tok

    async def option_probs(self, state: str, question: str, options: list[str]) -> list[float]:
        """Fallback decision engine: LLM next-token probabilities over lettered options."""
        letters = "ABCDEFGH"[: len(options)]
        prompt = (f"{state}\n\nQuestion: {question}\n" + "\n".join(f"{l}. {o}" for l, o in zip(letters, options)) +
                  "\nAnswer with a single letter.")
        body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 1, "temperature": 0,
                "logprobs": True, "top_logprobs": 10}
        r = await self.client.post("/v1/chat/completions", json=body, timeout=5.0)
        d = r.json()
        top = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        import math
        probs = {l: 0.0 for l in letters}
        for t in top:
            k = t["token"].strip().upper()[:1]
            if k in probs:
                probs[k] += math.exp(t["logprob"])
        s = sum(probs.values()) or 1.0
        return [probs[l] / s for l in letters]


class SystemOne:
    """Typed decisions with calibrated probabilities (Kev-4B at /v1/systemone); LLM fallback."""

    def __init__(self, llm: LLM):
        self.client = httpx.AsyncClient(base_url=KEV_URL, timeout=httpx.Timeout(15.0, connect=0.5))
        self.llm = llm
        self.ok = False
        self.engine = "kev"

    async def health(self) -> bool:
        try:
            r = await self.client.get("/v1/models", timeout=1.0)
            self.ok = r.status_code == 200
        except Exception:
            self.ok = False
        return self.ok

    async def decide(self, decisions: list[dict]) -> dict:
        t = time.perf_counter()
        items = []
        engine = "kev-4b"
        try:
            if not self.ok and not await self.health():
                raise RuntimeError("kev down")
            # One request per distinct state (Kev caches the state prefix and answers all questions in one pass).
            async def one(d):
                q = ({"type": "noul", "instructions": d["question"]} if d["type"] == "noul" else
                     {"type": "choice", "instructions": d["question"], "criteria": d["options"]})
                r = await self.client.post("/v1/systemone", json={"state": d["state"], "questions": {d["key"]: q}})
                r.raise_for_status()
                return d, r.json()
            res = await asyncio.gather(*(one(d) for d in decisions))
            for d, js in res:
                items.append(_parse_kev(d, js))
        except Exception:
            engine = "llm-probs"
            items = []
            for d in decisions:
                if d["type"] == "noul":
                    pr = await self.llm.option_probs(d["state"], d["question"], ["Yes", "No"])
                    items.append({"label": d.get("label") or d["question"], "question": d["question"], "type": "noul",
                                  "p_yes": pr[0]})
                else:
                    names = list(d["options"])
                    pr = await self.llm.option_probs(d["state"], d["question"], names)
                    items.append({"label": d["question"], "question": d["question"], "type": "choice",
                                  "options": [{"name": n, "p": p} for n, p in sorted(zip(names, pr), key=lambda x: -x[1])]})
        return {"engine": engine, "ms": (time.perf_counter() - t) * 1000, "items": items}


def _parse_kev(d: dict, js: dict) -> dict:
    a = (js.get("answers") or {}).get(d["key"]) or {}
    if d["type"] == "noul":
        return {"label": d.get("label") or d["question"], "question": d["question"], "type": "noul",
                "p_yes": float(a.get("noul", 0.5)), "server_ms": js.get("latency_ms")}
    probs = a.get("probabilities") or {}
    opts = [{"name": k, "p": float(v)} for k, v in sorted(probs.items(), key=lambda kv: -kv[1])]
    return {"label": d["question"], "question": d["question"], "type": "choice", "options": opts,
            "server_ms": js.get("latency_ms")}


ABBREV = re.compile(r"(?:\b(?:Dr|Mr|Ms|Mrs|Prof|St|vs|Inc|Co|No)|\be\.g|\bi\.e)\.$")


def split_sentences(buf: str) -> tuple[list[str], str]:
    raw = SENT_END.split(buf)
    parts = []
    for p in raw:  # "Dr. Jordan Ashdown" is not a sentence break
        if parts and ABBREV.search(parts[-1].rstrip()):
            parts[-1] = parts[-1].rstrip() + " " + p
        else:
            parts.append(p)
    if len(parts) == 1:
        return [], buf
    return [" ".join(p.split()) for p in parts[:-1] if p.strip()], parts[-1]


def is_follow_up(question: str) -> bool:
    q = question.lower().strip(" ?.!")
    return len(q.split()) <= 6 and bool(re.match(r"^(why|how come|and|what about|how about|tell me more|more|explain|"
                                                  r"what does that mean|so what|which one|what else)\b", q))


async def answer(question: str, persona: str, llm: LLM, use_llm: bool = True, lang: str = "en",
                 first_audio_ready: asyncio.Event | None = None, previous: str | None = None) -> AsyncIterator[dict]:
    """Yields events: route, result, sentence*, done."""
    STATS["questions"] += 1
    timings = {}
    t0 = time.perf_counter()
    context = previous if (previous and is_follow_up(question)) else None
    r_tool, r_args, r_conf = route(question if not context else f"{context} {question}", persona)
    tool, args, picked_by = r_tool, r_args, "router"
    if use_llm and llm.ok:
        try:
            l_tool, l_args, meta = await llm.pick_tool(question, persona, context)
            timings["llm_route_ms"] = meta["ms"]
            if l_tool == "search_knowledge" and r_tool != "search_knowledge" and r_conf >= 0.9:
                l_tool = None  # an analytics question the router is sure about stays with its analytics tool
            if r_tool == "search_knowledge" and l_tool and l_tool != "search_knowledge":
                l_tool = None  # explicit document question ("what did the QBR say") goes to the documents
            if l_tool:
                tool, args, picked_by = l_tool, l_args, "llm"
                # The router extracts literal arguments (cities, stage, points) more reliably than a 3B-active model.
                if l_tool == r_tool and r_args:
                    args = {**l_args, **r_args}
        except Exception as e:  # noqa: BLE001
            timings["llm_route_error"] = str(e)[:80]
    timings["route_ms"] = (time.perf_counter() - t0) * 1000
    yield {"type": "route", "tool": tool, "args": args, "by": picked_by, "router_tool": r_tool,
           "agree": tool == r_tool, "ms": timings.get("llm_route_ms", timings["route_ms"])}

    if tool == "search_knowledge":
        args = {"question": question if not context else f"{context} {question}"}
    t1 = time.perf_counter()
    try:
        result = run_tool(persona, tool, args)
    except Exception:
        tool, args = r_tool, r_args
        result = run_tool(persona, tool, args)
    timings["tool_ms"] = (time.perf_counter() - t1) * 1000
    yield {"type": "result", "result": result, "ms": timings["tool_ms"]}
    async for ev in narrate_result(question, persona, result, llm, use_llm, lang, first_audio_ready, context,
                                   timings, t0, tool, picked_by):
        yield ev


async def narrate_result(question: str, persona: str, result: dict, llm: LLM, use_llm: bool, lang: str,
                         first_audio_ready: asyncio.Event | None, context: str | None, timings: dict, t0: float,
                         tool: str, picked_by: str, already_spoken: list[str] | None = None) -> AsyncIterator[dict]:
    """Narrate a tool result sentence by sentence; every sentence passes the grounding check or is cut."""
    t2 = time.perf_counter()
    spoken: list[str] = []
    checked = []
    mode = "llm"
    failed = None
    if use_llm and llm.ok:
        buf = ""
        try:
            ctl = {"pause": False, "resume": first_audio_ready}
            stream = llm.narrate_paused(llm.narrate_messages(question, persona, result, lang, context), ctl,
                                        prefix="Según el análisis, " if lang == "es" else "")
            async for tok in stream:
                if "first_token_ms" not in timings:
                    timings["first_token_ms"] = (time.perf_counter() - t2) * 1000
                buf += tok
                sents, buf = split_sentences(buf)
                if not spoken and not sents:
                    # Start speaking at the first clause boundary so long opening sentences don't delay audio.
                    k = buf.rfind(", ", 18)
                    if k > 0 and len(buf) > 40:
                        sents, buf = [" ".join(buf[:k + 1].split())], buf[k + 2:]
                stop = False
                for s in sents:
                    g = grounding.check(s, result)
                    if not g["ok"]:
                        failed = {"sentence": s, "unmatched": g["unmatched"]}
                        stop = True
                        break
                    checked.extend(g["numbers"])
                    spoken.append(s)
                    if len(spoken) == 1:
                        timings["first_sentence_ms"] = (time.perf_counter() - t2) * 1000
                        ctl["pause"] = first_audio_ready is not None
                    yield {"type": "sentence", "text": s, "source": "llm"}
                    # Voice answers stay short: stop after 3 full sentences or ~60 words.
                    full = sum(1 for x in spoken if x.rstrip().endswith((".", "!", "?")))
                    if full >= 3 or sum(len(x.split()) for x in spoken) >= 60:
                        stop = True
                        buf = ""
                        break
                if stop:
                    break
            # A leftover that ends in "digit." was cut by the token limit mid-number (e.g. "$15."), not a sentence.
            if not failed and buf.strip() and buf.strip().endswith((".", "!", "?")) and not re.search(r"\d\.$", buf.strip()):
                s = " ".join(buf.split())
                g = grounding.check(s, result)
                if g["ok"]:
                    checked.extend(g["numbers"])
                    spoken.append(s)
                    if len(spoken) == 1:
                        timings["first_sentence_ms"] = (time.perf_counter() - t2) * 1000
                    yield {"type": "sentence", "text": s, "source": "llm"}
                else:
                    failed = {"sentence": s, "unmatched": g["unmatched"]}
        except Exception as e:  # noqa: BLE001
            timings["narrate_error"] = str(e)[:80]
    if not spoken:
        mode = "template"
        sents, rest = split_sentences(result["template"] + " ")
        for s in sents + ([rest.strip()] if rest.strip() else []):
            g = grounding.check(s, result)
            checked.extend(g["numbers"])
            spoken.append(s)
            if len(spoken) == 1:
                timings["first_sentence_ms"] = (time.perf_counter() - t2) * 1000
            yield {"type": "sentence", "text": s, "source": "template"}
    elif failed:
        mode = "llm+trimmed"
    timings["narrate_ms"] = (time.perf_counter() - t2) * 1000
    timings["total_ms"] = (time.perf_counter() - t0) * 1000
    n_ok = sum(1 for c in checked if c["matched"])
    if failed:
        STATS["grounding_fail"] += 1
    else:
        STATS["grounding_pass"] += 1
    grounded = {"numbers": len(checked), "verified": n_ok, "failed_sentence": failed, "mode": mode,
                "blocked_numbers": len(failed["unmatched"]) if failed else 0}
    STATS["last"] = {"tool": tool, "timings": timings, "grounded": grounded}
    text = " ".join((already_spoken or []) + spoken).rstrip()
    if text.endswith((",", ";", ":")):  # an answer trimmed right after a clause still ends as a sentence
        text = text[:-1] + "."
    yield {"type": "done", "text": text, "timings": timings, "grounding": grounded, "tool": tool,
           "picked_by": picked_by}
