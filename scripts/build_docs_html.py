from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "PROJECT_GUIDE.md"
OUT = ROOT / "docs" / "PROJECT_GUIDE.html"

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GraphRAG &amp; Self-Corrective Hybrid Retrieval Engine — Project Guide</title>
<style>
:root { --bg:#0b1220; --panel:#111a2e; --panel2:#16213b; --line:#24304d;
        --text:#dbe4f3; --muted:#8fa1c0; --accent:#8b5cf6; --accent2:#38bdf8; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font:15px/1.65 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; }
a { color:var(--accent2); text-decoration:none; } a:hover { text-decoration:underline; }
header { background:linear-gradient(120deg,#0b1220 0%,#1e3a8a 55%,#7c3aed 100%);
         padding:2.2rem 2rem 1.6rem; border-bottom:1px solid var(--line); }
header h1 { margin:0; font-size:1.55rem; color:#fff; }
header p { margin:.5rem 0 0; color:#c7d2fe; font-size:.95rem; max-width:900px; }
.badges { margin-top:.9rem; }
.badges span { display:inline-block; background:rgba(255,255,255,.12); color:#e0e7ff;
               border:1px solid rgba(255,255,255,.25); border-radius:999px;
               padding:.15rem .7rem; font-size:.72rem; margin:.15rem .3rem 0 0; }
.wrap { display:grid; grid-template-columns:250px 1fr; gap:0; max-width:1280px; margin:0 auto; }
nav { position:sticky; top:0; align-self:start; max-height:100vh; overflow:auto;
      padding:1.4rem 1rem; border-right:1px solid var(--line); font-size:.86rem; }
nav .toc-title { color:var(--muted); text-transform:uppercase; letter-spacing:.08em;
                 font-size:.7rem; margin-bottom:.6rem; }
nav ul { list-style:none; margin:0; padding:0; }
nav li { margin:.28rem 0; }
nav a { color:var(--muted); } nav a:hover { color:var(--accent2); }
nav ul ul { padding-left:.9rem; }
main { padding:2rem 2.4rem 4rem; min-width:0; }
h1 { font-size:1.45rem; }
h2 { font-size:1.15rem; color:#fff; border-bottom:2px solid var(--accent);
     padding-bottom:.35rem; margin-top:2.6rem; }
h3 { font-size:1rem; color:var(--accent2); margin-top:1.8rem; }
p, li { color:var(--text); }
strong { color:#fff; }
blockquote { border-left:4px solid var(--accent); margin:1rem 0; padding:.6rem 1rem;
             background:var(--panel); border-radius:0 8px 8px 0; color:var(--muted); }
code { background:var(--panel2); border:1px solid var(--line); border-radius:5px;
       padding:.08rem .35rem; font-size:.85em; color:#c4b5fd;
       font-family:'SF Mono',Menlo,Consolas,monospace; }
pre { background:var(--panel2); border:1px solid var(--line); border-radius:10px;
      padding:1rem; overflow-x:auto; }
pre code { background:none; border:none; padding:0; color:var(--text); }
table { border-collapse:collapse; width:100%; margin:1rem 0; font-size:.86rem; }
th { background:var(--panel2); color:#fff; text-align:left; }
th, td { border:1px solid var(--line); padding:.5rem .7rem; vertical-align:top; }
tr:nth-child(even) td { background:rgba(255,255,255,.02); }
hr { border:none; border-top:1px solid var(--line); margin:2.4rem 0; }
.mermaid { background:var(--panel); border:1px solid var(--line); border-radius:12px;
           padding:1rem; margin:1.2rem 0; }
footer { border-top:1px solid var(--line); color:var(--muted); font-size:.8rem;
         padding:1.2rem 2.4rem; }
@media (max-width:900px){ .wrap { grid-template-columns:1fr; } nav { position:static;
  max-height:none; border-right:none; border-bottom:1px solid var(--line); } }
@media print {
  body { background:#fff; color:#111; }
  header { background:#1e3a8a !important; -webkit-print-color-adjust:exact; }
  nav, footer { display:none; }
  .wrap { grid-template-columns:1fr; }
  pre, table, blockquote { background:#f5f5f5; border-color:#ddd; }
  pre code, code { color:#111; background:none; border:none; }
  h2 { color:#111; } th { color:#111; } strong { color:#111; }
}
</style>
</head>
<body>
<header>
  <h1>🕸️ GraphRAG &amp; Self-Corrective Hybrid Retrieval Engine — Project Guide</h1>
  <p>Neo4j knowledge graph · Qdrant dense+sparse · RRF fusion · cross-encoder rerank ·
     CRAG loop with web fallback · Ragas/DeepEval continuous evaluation — built entirely on
     free, self-hostable resources.</p>
  <div class="badges">
    <span>Python 3.11</span><span>FastAPI</span><span>Neo4j 5</span><span>Qdrant</span>
    <span>BGE-M3</span><span>LangGraph</span><span>Instructor</span><span>Groq gpt-oss</span>
    <span>Ragas</span><span>DeepEval</span><span>Streamlit</span><span>Celery + Redis</span>
  </div>
</header>
<div class="wrap">
<nav id="toc"><div class="toc-title">Contents</div>__TOC__</nav>
<main>__BODY__</main>
</div>
<footer>Generated from docs/PROJECT_GUIDE.md · every metric measured on the running engine ·
rebuild with: .venv/bin/python scripts/build_docs_html.py</footer>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>mermaid.initialize({ startOnLoad: true, theme: "dark", flowchart: { curve: "basis" } });</script>
</body>
</html>
"""


def main() -> None:
    from markdown import Markdown

    md = Markdown(
        extensions=["tables", "fenced_code", "toc", "sane_lists", "smarty"],
        extension_configs={"toc": {"toc_depth": "2-3"}},
    )
    body = md.convert(SRC.read_text(encoding="utf-8"))
    toc = md.toc

    body = re.sub(
        r'<pre><code class="language-mermaid">(.*?)</code></pre>',
        r'<div class="mermaid">\1</div>',
        body,
        flags=re.S,
    )
    html = TEMPLATE.replace("__TOC__", toc).replace("__BODY__", body)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
