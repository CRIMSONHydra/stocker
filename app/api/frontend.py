"""Serves the embedded HTML frontend.

When a built React SPA exists at ``frontend/dist/index.html``, that is served at
``/`` and ``/web``. Otherwise the inline HTML below is served as a fallback so
the API is usable without Node tooling (and so tests do not require a build).
"""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse

router = APIRouter(tags=["frontend"])

DIST_INDEX = Path(__file__).resolve().parents[2] / "frontend" / "dist" / "index.html"

FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Stocker - OpenEnv</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }
  .container { max-width: 900px; margin: 0 auto; }
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
  <h1>Stocker</h1>
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


@router.get("/")
@router.get("/web")
async def index():
    if DIST_INDEX.is_file():
        return FileResponse(DIST_INDEX)
    return HTMLResponse(content=FRONTEND_HTML)
