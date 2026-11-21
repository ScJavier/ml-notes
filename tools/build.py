#!/usr/bin/env python3
"""Convierte src/*.md en fragmentos HTML (KaTeX ya renderizado) + notes/index.json.
Uso: cd tools && npm install && python3 build.py"""
import re, json, html, subprocess, os, hashlib, glob, base64, io
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src"); OUT = os.path.join(ROOT, "notes")
os.makedirs(OUT, exist_ok=True)
pat = re.compile(r'<span class="math (inline|display)">\\(?:\(|\[)(.*?)\\(?:\)|\])</span>', re.S)

def build(path):
    text = open(path, encoding="utf-8").read()
    m = re.match(r"---\n(.*?)\n---\n(.*)", text, re.S)
    meta = dict(re.findall(r'^(\w+):\s*"?(.*?)"?\s*$', m.group(1), re.M)); md = m.group(2)
    body = subprocess.run(["pandoc", "-f", "markdown+tex_math_dollars", "-t", "html5", "--mathjax", "--wrap=none"],
                          input=md, capture_output=True, text=True, check=True).stdout
    items = [(html.unescape(x.group(2)).strip(), x.group(1) == "display") for x in pat.finditer(body)]
    json.dump(items, open(os.path.join(HERE, "m_in.json"), "w"))
    subprocess.run(["node", os.path.join(HERE, "render_math.js"), os.path.join(HERE, "m_in.json"), os.path.join(HERE, "m_out.json")], check=True, cwd=HERE)
    out = iter(json.load(open(os.path.join(HERE, "m_out.json"))))
    body = pat.sub(lambda x: next(out), body)
    body = re.sub(r'<img src="(fig/[^"]+)"', lambda m_: '<img src="%s" width="%d" height="%d"' % incrustar(os.path.join(SRC, m_.group(1))), body)
    return meta, body

def incrustar(ruta):
    """Imagen como data URI en WebP: la nota queda en un solo archivo y funciona sin conexión."""
    im = Image.open(ruta).convert("RGB"); buf = io.BytesIO()
    im.save(buf, "WEBP", quality=82, method=6)
    return ("data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode(),) + im.size

index = []
for p in sorted(glob.glob(os.path.join(SRC, "*.md"))):
    nid = re.sub(r"^\d+-", "", os.path.splitext(os.path.basename(p))[0])
    meta, body = build(p)
    open(os.path.join(OUT, nid + ".html"), "w", encoding="utf-8").write(body)
    index.append({"id": nid, "title": meta.get("title", nid), "author": meta.get("author", ""),
                  "file": f"notes/{nid}.html", "rev": hashlib.md5(body.encode()).hexdigest()[:8]})
    print("nota:", nid, len(body) // 1024, "KB")
json.dump(index, open(os.path.join(OUT, "index.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
