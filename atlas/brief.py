"""Executive Brief: a branded 1080x1350 PNG and a one-page PDF built from the last answer. Fully local."""
from __future__ import annotations

import datetime as dt
import textwrap

from PIL import Image, ImageDraw, ImageFont

from atlas.analytics.common import fmt
from atlas.config import ASSISTANT_NAME, OUT_DIR, PERSONAS

BLUE = (0, 75, 223)
BLOCK = (209, 221, 241)
META = (140, 140, 140)
INK = (0, 0, 0)
SOFT = (128, 165, 239)
# Liberation Sans is metric-compatible with Arial and ships with the OS (Arial itself is not installed).
FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
FONT_B = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"


def _f(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def _label(d, xy, text, size=22):
    x, y = xy
    for ch in text.upper():
        d.text((x, y), ch, font=_f(size, True), fill=BLUE)
        x += d.textlength(ch, font=_f(size, True)) + size * 0.14


def _wrap(d, text, font, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _chart(d, box, chart):
    x0, y0, x1, y1 = box
    labels = chart["labels"]
    series = chart["series"]
    main = series[-1]["data"] if chart["type"] != "hbar" else series[0]["data"]
    kind = chart.get("fmt", "num")
    n = len(labels)
    if chart["type"] == "hbar":
        vmax = max(max(s["data"]) for s in series) or 1
        row = min((y1 - y0) / n, 66)
        for i, lab in enumerate(labels):
            y = y0 + i * row
            d.text((x0, y + row * 0.45 - 13), textwrap.shorten(lab, 24, placeholder="…"), font=_f(24, True), fill=INK)
            bx0 = x0 + 330
            if len(series) > 1:
                w = (x1 - bx0 - 150) * series[1]["data"][i] / vmax
                d.rectangle([bx0, y + row * 0.15, bx0 + w, y + row * 0.75], fill=(201, 214, 238))
            w = (x1 - bx0 - 150) * main[i] / vmax
            d.rectangle([bx0, y + row * 0.15, bx0 + w, y + row * 0.75], fill=BLUE)
            d.text((bx0 + w + 10, y + row * 0.45 - 12), fmt(main[i], kind), font=_f(22, True), fill=INK)
        return
    # vertical bars
    vals = [v for s in series for v in s["data"]] + ([chart["band"]["high"]] if chart.get("band") else [])
    vmax = max(vals) * 1.1 or 1
    vmin = 0
    base_y = y1 - 46
    d.line([x0, base_y, x1, base_y], fill=INK, width=2)
    groups = len(series)
    slot = (x1 - x0) / n
    bw = slot * 0.7 / groups

    def ypix(v):
        return base_y - (v - vmin) / (vmax - vmin) * (base_y - y0 - 30)

    for i, lab in enumerate(labels):
        for gi, s in enumerate(series):
            v = s["data"][i]
            bx = x0 + i * slot + slot * 0.15 + gi * bw
            last = gi == groups - 1
            color = BLUE if (last and (not chart.get("highlight_last") or i == n - 1)) else (SOFT if last else (159, 180, 223))
            d.rectangle([bx, ypix(v), bx + bw - 4, base_y], fill=color)
            if last:
                t = fmt(v, kind)
                tw = d.textlength(t, font=_f(20, True))
                top = ypix(v)
                if chart.get("band") and i == n - 1:
                    top = min(top, ypix(chart["band"]["high"]) - 4)
                d.text((bx + (bw - 4 - tw) / 2, top - 28), t, font=_f(20, True), fill=INK)
        lab_s = textwrap.shorten(lab, 16, placeholder="…")
        tw = d.textlength(lab_s, font=_f(19, True))
        d.text((x0 + i * slot + (slot - tw) / 2, base_y + 10), lab_s, font=_f(19, True), fill=INK)
    if chart.get("band"):
        bx = x0 + (n - 1) * slot + slot * 0.15 + (groups - 1) * bw + (bw - 4) / 2
        ya, yb = ypix(chart["band"]["low"]), ypix(chart["band"]["high"])
        d.line([bx, ya, bx, yb], fill=INK, width=3)
        d.line([bx - 12, ya, bx + 12, ya], fill=INK, width=3)
        d.line([bx - 12, yb, bx + 12, yb], fill=INK, width=3)


def build_brief(last: dict, qr_url: str = "") -> dict:
    r = last["result"]
    persona = PERSONAS[last["persona"]]
    W, H = 1080, 1350
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    # header
    d.rectangle([0, 0, W, 14], fill=BLUE)
    d.text((60, 48), ASSISTANT_NAME.upper(), font=_f(44, True), fill=BLUE)
    d.text((60 + d.textlength(ASSISTANT_NAME.upper(), font=_f(44, True)) + 24, 64), "EXECUTIVE BRIEF", font=_f(22, True), fill=META)
    d.text((60, 108), persona["org"], font=_f(28, True), fill=INK)
    d.text((W - 60 - d.textlength(dt.date.today().strftime("%d %b %Y"), font=_f(22)), 60), dt.date.today().strftime("%d %b %Y"),
           font=_f(22), fill=META)
    # question
    _label(d, (60, 170), "Question")
    y = 204
    for line in _wrap(d, last["question"], _f(34, True), W - 120)[:3]:
        d.text((60, y), line, font=_f(34, True), fill=INK)
        y += 44
    # headline block
    y += 16
    d.rectangle([60, y, W - 60, y + 230], fill=BLOCK)
    _label(d, (88, y + 24), r["headline"]["label"])
    hv = r["headline"]["value"]
    htxt = hv if isinstance(hv, str) else fmt(hv, r["headline"]["fmt"])
    size = 108 if len(htxt) <= 10 else (70 if len(htxt) <= 20 else 50)
    d.text((88, y + 58), htxt, font=_f(size, True), fill=INK)
    for i, line in enumerate(_wrap(d, r["headline"].get("sub", ""), _f(24), W - 176)[:2]):
        d.text((88, y + 176 + i * 28), line, font=_f(24), fill=INK)
    y += 262
    # chart (shorter for horizontal bars with few rows)
    ch = 330 if r["chart"]["type"] != "hbar" else min(330, 66 * len(r["chart"]["labels"]) + 10)
    if r["chart"].get("series") and r["chart"]["type"] != "none":
        _label(d, (60, y), r["chart"]["title"])
        _chart(d, (60, y + 40, W - 60, y + 40 + ch), r["chart"])
    else:
        # No chart (document answers, developer runs, help): show the cited passages or the real result lines instead.
        items = ([f"{c['title']}: {c['text']}" for c in r.get("citations", [])] or r.get("facts", []))[:4]
        _label(d, (60, y), "Cited passages (synthetic documents)" if r.get("citations") else "Result")
        yy = y + 44
        for it in items:
            for line in _wrap(d, it, _f(22), W - 120)[:4]:
                if yy > y + 40 + ch:
                    break
                d.text((60, yy), line, font=_f(22), fill=INK)
                yy += 28
            yy += 10
    y += ch + 64
    text_w = W - 120 - (200 if qr_url else 0)
    bottom = H - 96 - 24
    # answer: as many lines as fit, keeping room for the method block (~120 px)
    _label(d, (60, y), "Answer")
    y += 34
    lines = _wrap(d, last["answer"], _f(25), text_w)
    max_lines = max(2, min(5, int((bottom - 130 - y) // 33)))
    for k, line in enumerate(lines[:max_lines]):
        if k == max_lines - 1 and len(lines) > max_lines:
            line = line.rstrip(".,; ") + "…"
        d.text((60, y), line, font=_f(25), fill=INK)
        y += 33
    # method
    y += 12
    _label(d, (60, y), "How we got this")
    y += 32
    for line in _wrap(d, r["method"]["formula"], _f(20), text_w)[:2]:
        d.text((60, y), line, font=_f(20), fill=INK)
        y += 26
    g = last.get("grounding") or {}
    d.text((60, min(y + 6, bottom - 4)), f"{g.get('verified', 0)}/{g.get('numbers', 0)} numbers verified against deterministic tool output",
           font=_f(19, True), fill=(14, 138, 59))
    # QR, bottom-right above the footer, outside the text column
    if qr_url:
        import qrcode
        q = qrcode.QRCode(border=1, box_size=5)
        q.add_data(qr_url)
        q.make(fit=True)
        qi = q.make_image(fill_color="black", back_color="white").convert("RGB").resize((170, 170))
        img.paste(qi, (W - 60 - 170, H - 96 - 24 - 170))
    # footer
    d.rectangle([0, H - 96, W, H], fill=BLUE)
    d.text((60, H - 78), "Generated on HP ZGX Nano, fully offline.", font=_f(28, True), fill="white")
    d.text((60, H - 40), "All data synthetic. Fictional companies.", font=_f(19), fill="white")
    OUT_DIR.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    png = OUT_DIR / f"brief-{stamp}.png"
    pdf = OUT_DIR / f"brief-{stamp}.pdf"
    img.save(png, optimize=True)
    _pdf(img, pdf, last)
    return {"png": f"/out/{png.name}", "pdf": f"/out/{pdf.name}"}


def _pdf(img: Image.Image, path, last: dict):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(path), pagesize=letter)
    pw, ph = letter
    h = ph - 72
    w = h * img.width / img.height
    c.drawImage(ImageReader(img), (pw - w) / 2, 36, width=w, height=h)
    c.setTitle(f"{ASSISTANT_NAME} Executive Brief")
    c.setAuthor(f"{ASSISTANT_NAME} on HP ZGX Nano (offline)")
    c.showPage()
    c.save()
