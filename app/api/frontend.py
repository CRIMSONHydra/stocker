"""Frontend routes.

- ``/``    — the README rendered as a blog-style landing page (judges read first)
- ``/web`` — the React SPA (or inline HTML fallback if no built dist)
- ``/blog`` — alias of ``/``
"""

import re
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse
from markdown_it import MarkdownIt

router = APIRouter(tags=["frontend"])

ROOT = Path(__file__).resolve().parents[2]
DIST_INDEX = ROOT / "frontend" / "dist" / "index.html"
README_PATH = ROOT / "README.md"

GITHUB_BLOB = "https://github.com/CRIMSONHydra/stocker/blob/main/"
GITHUB_RAW  = "https://raw.githubusercontent.com/CRIMSONHydra/stocker/main/"


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[end + 5:].lstrip()
    return text


def _rewrite_urls(html: str) -> str:
    """Rewrite relative href/src to absolute GitHub URLs and force external
    links to open in a new tab (the HF Space iframe blocks in-frame nav to
    cross-origin URLs, so plain <a> clicks silently fail)."""
    # 1) Rewrite relative href to GitHub blob URL
    def fix_href(m: re.Match) -> str:
        url = m.group(1)
        if url.startswith(("http://", "https://", "/", "#", "mailto:")):
            return m.group(0)
        return f'href="{GITHUB_BLOB}{url}"'
    html = re.sub(r'href="([^"#?][^"]*)"', fix_href, html)

    # 2) Rewrite relative img src to GitHub raw URL
    def fix_src(m: re.Match) -> str:
        url = m.group(1)
        if url.startswith(("http://", "https://", "/")):
            return m.group(0)
        return f'src="{GITHUB_RAW}{url}"'
    html = re.sub(r'src="([^"#?][^"]*)"', fix_src, html)

    # 3) Add target="_blank" + rel to every external <a href="http(s)://...">
    #    so clicks escape the HF Space iframe.
    def add_target(m: re.Match) -> str:
        tag = m.group(0)
        if "target=" in tag:
            return tag
        return tag[:-1] + ' target="_blank" rel="noopener noreferrer">'
    html = re.sub(r'<a\s+href="https?://[^"]+"\s*>', add_target, html)

    return html


def _render_readme() -> str:
    if not README_PATH.is_file():
        return "<p><em>README.md not bundled in this image.</em></p>"
    md_text = _strip_frontmatter(README_PATH.read_text(encoding="utf-8"))
    md = MarkdownIt("commonmark").enable(["table", "strikethrough"])
    return _rewrite_urls(md.render(md_text))


BLOG_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Stocker — Multi-Agent Council RL for Stock Trading</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; line-height: 1.7; }}
    .nav {{ background: rgba(15, 23, 42, 0.95); backdrop-filter: blur(8px); border-bottom: 1px solid #334155; padding: 14px 24px; position: sticky; top: 0; z-index: 100; }}
    .nav-inner {{ max-width: 920px; margin: 0 auto; display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }}
    .nav-title {{ font-weight: 700; color: #38bdf8; font-size: 1.05em; }}
    .nav-links a {{ color: #94a3b8; text-decoration: none; margin-left: 20px; font-size: 0.9em; transition: color 0.15s; }}
    .nav-links a:hover {{ color: #e2e8f0; }}
    .nav-links a.cta {{ background: #2563eb; color: white; padding: 7px 16px; border-radius: 8px; font-weight: 600; }}
    .nav-links a.cta:hover {{ background: #1d4ed8; color: white; }}
    article {{ max-width: 880px; margin: 0 auto; padding: 32px 24px 80px; }}
    article h1 {{ color: #38bdf8; font-size: 2.2em; margin: 0 0 0.5em; line-height: 1.2; }}
    article h2 {{ color: #38bdf8; margin: 1.6em 0 0.5em; border-top: 1px solid #1e293b; padding-top: 1em; font-size: 1.55em; }}
    article h3 {{ color: #cbd5e1; margin: 1.4em 0 0.4em; font-size: 1.2em; }}
    article a {{ color: #38bdf8; }}
    article a:hover {{ text-decoration: underline; }}
    article code {{ background: #1e293b; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; color: #fbbf24; }}
    article pre {{ background: #1e293b; padding: 16px; border-radius: 8px; overflow-x: auto; border: 1px solid #334155; }}
    article pre code {{ background: transparent; padding: 0; color: #e2e8f0; }}
    article table {{ border-collapse: collapse; margin: 1em 0; width: 100%; font-size: 0.9em; }}
    article th, article td {{ border: 1px solid #334155; padding: 8px 12px; text-align: left; }}
    article th {{ background: #1e293b; color: #38bdf8; }}
    article img {{ max-width: 100%; border-radius: 8px; margin: 1em 0; }}
    article blockquote {{ border-left: 3px solid #38bdf8; margin: 1em 0; padding: 12px 16px; background: #1e293b; border-radius: 4px; color: #cbd5e1; }}
    article blockquote p {{ margin: 0.4em 0; }}
    article ul, article ol {{ padding-left: 24px; }}
    article hr {{ border: none; border-top: 1px solid #334155; margin: 2em 0; }}
    .demo-cta {{ display: inline-block; background: #2563eb; color: white !important; padding: 14px 28px; border-radius: 10px; text-decoration: none; font-weight: 600; margin: 16px 0; transition: background 0.15s; }}
    .demo-cta:hover {{ background: #1d4ed8; }}
  </style>
</head>
<body>
  <nav class="nav">
    <div class="nav-inner">
      <a href="#top" class="nav-title" style="text-decoration:none;">⚡ Stocker</a>
      <span class="nav-links">
        <a href="#top">Top</a>
        <a href="https://github.com/CRIMSONHydra/stocker" target="_blank" rel="noopener">GitHub</a>
        <a href="https://github.com/CRIMSONHydra/stocker/blob/main/BLOG.md" target="_blank" rel="noopener">Blog</a>
        <a href="/web" class="cta">🚀 Live Demo</a>
      </span>
    </div>
  </nav>
  <article id="top">{readme}</article>
</body>
</html>"""


FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Stocker - OpenEnv (demo)</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }
  .container { max-width: 900px; margin: 0 auto; }
  .back { color: #38bdf8; text-decoration: none; font-size: 0.9em; }
  h1 { color: #38bdf8; }
  .card { background: #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 16px; border: 1px solid #334155; }
  label { display: block; color: #94a3b8; font-size: 0.85em; margin: 8px 0 4px; }
  select, input { width: 100%; padding: 8px; border-radius: 6px; border: 1px solid #475569; background: #0f172a; color: #e2e8f0; }
  .btn { padding: 10px 20px; border-radius: 8px; border: none; cursor: pointer; font-weight: 600; margin-top: 10px; margin-right: 8px; }
  .btn-primary { background: #2563eb; color: white; }
  .btn-success { background: #059669; color: white; }
  .row { display: flex; gap: 12px; }
  .row > * { flex: 1; }
  .log { background: #0f172a; border-radius: 8px; padding: 12px; font-family: monospace; font-size: 0.85em; max-height: 280px; overflow-y: auto; white-space: pre-wrap; color: #94a3b8; }
  .stat { color: #94a3b8; }
  .stat strong { color: #e2e8f0; }
</style>
</head>
<body>
<div class="container">
  <a href="/" class="back">← Back to methodology</a>
  <h1>Stocker — Live Demo</h1>
  <p>OpenEnv RL environment for stock-trading decisions.</p>

  <div class="card">
    <div class="row">
      <div>
        <label>Task</label>
        <select id="taskSelect"><option>Loading...</option></select>
      </div>
      <div style="display:flex;align-items:flex-end;">
        <button class="btn btn-primary" onclick="resetEnv()">Start</button>
      </div>
    </div>
  </div>

  <div class="card" id="obsCard" style="display:none;">
    <p class="stat">Ticker: <strong id="ticker"></strong> | Date: <strong id="date"></strong> | Price: <strong id="price"></strong></p>
    <p class="stat">Cash: <strong id="cash"></strong> | Position: <strong id="position"></strong> | Portfolio: <strong id="pv"></strong></p>
    <p class="stat">Step <strong id="step"></strong> of <strong id="total"></strong></p>
  </div>

  <div class="card" id="actCard" style="display:none;">
    <div class="row">
      <div>
        <label>Side</label>
        <select id="side"><option>buy</option><option>sell</option><option selected>hold</option></select>
      </div>
      <div>
        <label>Quantity</label>
        <input type="number" id="qty" min="0" value="0">
      </div>
    </div>
    <button class="btn btn-success" onclick="submit()">Submit</button>
  </div>

  <div class="card">
    <div class="log" id="log">Ready.</div>
  </div>
</div>

<script>
function log(m) { const el = document.getElementById('log'); el.textContent += '\\n' + m; el.scrollTop = el.scrollHeight; }

async function loadTasks() {
  const r = await fetch('/meta');
  const d = await r.json();
  const sel = document.getElementById('taskSelect');
  sel.innerHTML = '';
  for (const t of d.tasks) {
    const o = document.createElement('option');
    o.value = t; o.textContent = t;
    sel.appendChild(o);
  }
}
loadTasks();

function showObs(o) {
  document.getElementById('obsCard').style.display = 'block';
  document.getElementById('actCard').style.display = 'block';
  document.getElementById('ticker').textContent = o.ticker;
  document.getElementById('date').textContent = o.date;
  document.getElementById('price').textContent = o.price.toFixed(2);
  document.getElementById('cash').textContent = o.cash.toFixed(2);
  document.getElementById('position').textContent = o.position;
  document.getElementById('pv').textContent = o.portfolio_value.toFixed(2);
  document.getElementById('step').textContent = o.step_number;
  document.getElementById('total').textContent = o.total_steps;
}

async function resetEnv() {
  const tid = document.getElementById('taskSelect').value;
  const r = await fetch('/reset', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({task_id: tid}) });
  const d = await r.json();
  log('--- reset: ' + tid + ' ---');
  showObs(d.observation);
}

async function submit() {
  const a = { side: document.getElementById('side').value, quantity: parseInt(document.getElementById('qty').value || '0') };
  log('action: ' + JSON.stringify(a));
  const r = await fetch('/step', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(a) });
  const d = await r.json();
  log('reward: ' + d.reward.toFixed(4) + (d.done ? ' [DONE]' : ''));
  showObs(d.observation);
}
</script>
</body>
</html>"""


@router.get("/", response_class=HTMLResponse)
@router.get("/blog", response_class=HTMLResponse)
async def landing_blog():
    return HTMLResponse(content=BLOG_HTML.format(readme=_render_readme()))


@router.get("/web")
async def demo():
    if DIST_INDEX.is_file():
        return FileResponse(DIST_INDEX)
    return HTMLResponse(content=FRONTEND_HTML)
