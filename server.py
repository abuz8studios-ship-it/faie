"""
Synthetic Neocortex v2 — Full 8-Heresy API + Interactive Explorer.

All 1,651 Quranic roots × 8 heresies in one brain.

Endpoints:
  GET /                        — Interactive neocortex explorer
  GET /api/neocortex/{root}    — Full 8-heresy pipeline
  GET /api/compare/{root1}/{root2} — Root comparison
  GET /api/stats               — System statistics
  GET /api/roots               — List all roots (paginated)
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from neocortex import SyntheticNeocortex

app = FastAPI(title="Synthetic Neocortex v2", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

print("Loading Synthetic Neocortex v2 (all 8 heresies)...")
cortex = SyntheticNeocortex()
print("Neocortex ready.\n")


def to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@app.get("/api/neocortex/{root}")
def neocortex_query(root: str):
    result = cortex.query(root)
    return JSONResponse(to_json_safe(result))


@app.get("/api/compare/{root1}/{root2}")
def compare_roots(root1: str, root2: str):
    result = cortex.compare_roots(root1, root2)
    return JSONResponse(to_json_safe(result))


@app.get("/api/stats")
def stats():
    return cortex.stats()


@app.get("/api/roots")
def list_roots(limit: int = 50, offset: int = 0):
    roots = []
    for name, data in cortex.graph["nodes"].items():
        if data.get("type") == "root" and name in cortex.node_to_idx:
            roots.append({
                "root": name,
                "frequency": data.get("frequency", 0),
                "summary": (data.get("summary") or "")[:120],
            })
    roots.sort(key=lambda x: -x["frequency"])
    return {
        "total": len(roots),
        "offset": offset,
        "roots": roots[offset:offset + limit],
    }


@app.get("/", response_class=HTMLResponse)
def explorer():
    emb_np = cortex.embeddings.numpy()
    graph = cortex.graph

    root_nodes = []
    for name, data in graph["nodes"].items():
        if data.get("type") != "root" or name not in cortex.node_to_idx:
            continue
        idx = cortex.node_to_idx[name]
        e = emb_np[idx]
        freq = data.get("frequency", 0)
        root_nodes.append({
            "name": name,
            "x": float(e[0]), "y": float(e[1]),
            "norm": float(np.linalg.norm(e)),
            "freq": freq,
            "summary": (data.get("summary") or "")[:80],
        })

    root_nodes.sort(key=lambda x: -x["freq"])
    vis_nodes = root_nodes[:400]
    vis_names = {n["name"] for n in vis_nodes}

    vis_edges = []
    for edge in graph["edges"][:20000]:
        s, t = edge["source"], edge["target"]
        if s in vis_names and t in vis_names:
            si, ti = cortex.node_to_idx[s], cortex.node_to_idx[t]
            vis_edges.append({
                "x1": float(emb_np[si][0]), "y1": float(emb_np[si][1]),
                "x2": float(emb_np[ti][0]), "y2": float(emb_np[ti][1]),
                "rel": edge.get("relation", ""),
            })

    n_roots = sum(1 for d in graph["nodes"].values() if d.get("type") == "root")
    total_params = cortex.stats()["total_neural_params"]
    nodes_json = json.dumps(vis_nodes, ensure_ascii=False)
    edges_json = json.dumps(vis_edges, ensure_ascii=False)

    # HTML head + CSS (f-string for Python vars, {{}} for CSS braces)
    html_head = f"""<!DOCTYPE html>
<html lang="en" dir="ltr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Synthetic Neocortex v2</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#FAF8F5;color:#2A2520;font-family:'DM Sans',system-ui,sans-serif;overflow:hidden;display:flex}}
#disk{{flex:1;position:relative}}
#canvas{{display:block;width:100%;height:100vh;cursor:crosshair}}
#panel{{width:460px;height:100vh;overflow-y:auto;border-left:1px solid #E8E2D9;background:#fff;padding:24px;flex-shrink:0}}
#panel h1{{font-size:18px;color:#C8A55C;margin-bottom:4px;letter-spacing:0.02em}}
#panel .sub{{font-size:12px;color:#9A9080;margin-bottom:16px}}
#search{{width:100%;padding:10px 14px;border:2px solid #E8E2D9;border-radius:8px;font-size:15px;outline:none;background:#FAF8F5;color:#2A2520;font-family:inherit;margin-bottom:16px}}
#search:focus{{border-color:#C8A55C}}
#search::placeholder{{color:#B8AD9E}}
.heresy{{margin-bottom:14px;padding:12px;background:#FAF8F5;border-radius:8px;border-left:3px solid #C8A55C}}
.heresy .label{{font-size:11px;font-weight:700;color:#C8A55C;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px}}
.heresy .val{{font-size:13px;color:#4A4540;line-height:1.5}}
.heresy .mono{{font-family:Consolas,'Courier New',monospace;font-size:12px;color:#6A6560}}
.bar-row{{display:flex;align-items:center;gap:6px;margin:2px 0}}
.bar-label{{width:70px;font-size:11px;text-align:right;color:#6A6560}}
.bar-track{{flex:1;height:16px;background:#E8E2D9;border-radius:4px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:4px;transition:width 0.3s}}
.bar-val{{width:36px;font-size:11px;color:#6A6560}}
.ternary{{display:flex;gap:3px;margin:4px 0}}
.ternary .cell{{width:24px;height:24px;border-radius:4px;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700}}
.cell-pos{{background:#C8A55C;color:#fff}}
.cell-neg{{background:#6A6560;color:#fff}}
.cell-zero{{background:#E8E2D9;color:#B8AD9E}}
.mem-row{{display:flex;justify-content:space-between;padding:2px 0;font-size:12px;border-bottom:1px solid #F0EBE4}}
.mem-name{{font-size:14px}}
.mem-score{{color:#C8A55C;font-weight:600}}
#tooltip{{position:fixed;pointer-events:none;background:rgba(42,37,32,0.95);border:1px solid #C8A55C;border-radius:8px;padding:10px 14px;font-size:13px;display:none;z-index:100;max-width:300px;color:#FAF8F5}}
#tooltip .ar{{font-size:22px;color:#C8A55C}}
#tooltip .en{{color:#ccc;margin-top:2px;font-size:12px}}
#tooltip .freq{{color:#C8A55C;font-size:11px;margin-top:2px}}
#empty-state{{color:#B8AD9E;text-align:center;padding:40px 20px;font-size:14px;line-height:1.6}}
#loading{{color:#C8A55C;font-style:italic;display:none}}
</style>
</head>
<body>
<div id="disk"><canvas id="canvas"></canvas></div>
<div id="panel">
  <h1>Synthetic Neocortex v2</h1>
  <p class="sub">{n_roots} roots &middot; 8 heresies &middot; {total_params:,} params</p>
  <input id="search" type="text" placeholder="Search a root..." autofocus />
  <div id="loading">Querying neocortex...</div>
  <div id="results">
    <div id="empty-state">Click a node on the Poincar&eacute; disk<br>or type a root to explore all 8 heresies.</div>
  </div>
</div>
<div id="tooltip"><div class="ar"></div><div class="en"></div><div class="freq"></div></div>
<script>
const NODES = {nodes_json};
const EDGES = {edges_json};
"""

    # JavaScript app code — plain string, NO f-string (backticks + braces are literal)
    js_app = r"""
const DOMAIN_COLORS = {
  theology: '#C8A55C', nature: '#4CAF50', law: '#2196F3', narrative: '#E74C3C'
};
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');
const searchBox = document.getElementById('search');
const resultsDiv = document.getElementById('results');
const loadingDiv = document.getElementById('loading');
let W, H, cx, cy, R, scale = 1.0, hoveredNode = null, selectedNode = null;

function resize() {
  const rect = canvas.parentElement.getBoundingClientRect();
  W = rect.width; H = rect.height;
  canvas.width = W * devicePixelRatio;
  canvas.height = H * devicePixelRatio;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  cx = W / 2; cy = H / 2;
  R = Math.min(W, H) * 0.4;
}

function toScreen(x, y) { return [cx + x * R * scale * 2.5, cy + y * R * scale * 2.5]; }

function draw() {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#FAF8F5'; ctx.fillRect(0, 0, W, H);
  ctx.beginPath(); ctx.arc(cx, cy, R * scale, 0, Math.PI * 2);
  ctx.strokeStyle = '#E8E2D9'; ctx.lineWidth = 2; ctx.stroke();

  for (const e of EDGES) {
    const [x1, y1] = toScreen(e.x1, e.y1);
    const [x2, y2] = toScreen(e.x2, e.y2);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    ctx.strokeStyle = e.rel === 'semantic' ? 'rgba(200,165,92,0.06)' : 'rgba(0,0,0,0.03)';
    ctx.lineWidth = 0.5; ctx.stroke();
  }

  for (const n of NODES) {
    const [sx, sy] = toScreen(n.x, n.y);
    const isHov = hoveredNode === n;
    const isSel = selectedNode === n.name;
    const fscale = Math.min(1 + Math.log10(Math.max(n.freq, 1)) / 3, 2.5);
    const r = (isHov || isSel ? 6 : 4) * fscale;
    ctx.beginPath(); ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fillStyle = isSel ? '#C8A55C' : (isHov ? '#2A2520' : 'rgba(200,165,92,0.6)');
    ctx.fill();
    if (n.freq > 150 && scale > 0.5) {
      ctx.font = (10 + fscale) + 'px system-ui';
      ctx.fillStyle = 'rgba(42,37,32,0.5)';
      ctx.textAlign = 'center';
      ctx.fillText(n.name, sx, sy - r - 4);
    }
  }
  requestAnimationFrame(draw);
}

canvas.addEventListener('mousemove', (e) => {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  hoveredNode = null;
  for (const n of NODES) {
    const [sx, sy] = toScreen(n.x, n.y);
    if ((mx-sx)**2 + (my-sy)**2 < 400) { hoveredNode = n; break; }
  }
  if (hoveredNode) {
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX + 15) + 'px';
    tooltip.style.top = (e.clientY - 10) + 'px';
    tooltip.querySelector('.ar').textContent = hoveredNode.name;
    tooltip.querySelector('.en').textContent = hoveredNode.summary || '';
    tooltip.querySelector('.freq').textContent = hoveredNode.freq ? 'freq: ' + hoveredNode.freq : '';
    canvas.style.cursor = 'pointer';
  } else {
    tooltip.style.display = 'none';
    canvas.style.cursor = 'crosshair';
  }
});

canvas.addEventListener('click', () => { if (hoveredNode) queryRoot(hoveredNode.name); });
canvas.addEventListener('wheel', (e) => {
  e.preventDefault(); scale *= e.deltaY > 0 ? 0.9 : 1.1;
  scale = Math.max(0.2, Math.min(8, scale));
});
searchBox.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { const q = searchBox.value.trim(); if (q) queryRoot(q); }
});

async function queryRoot(name) {
  selectedNode = name;
  searchBox.value = name;
  loadingDiv.style.display = 'block';
  resultsDiv.innerHTML = '';
  try {
    const resp = await fetch('/api/neocortex/' + encodeURIComponent(name));
    const data = await resp.json();
    loadingDiv.style.display = 'none';
    if (data.error) { resultsDiv.innerHTML = '<div id="empty-state">' + data.error + '</div>'; return; }
    const h = data.heresies;
    let html = '';

    var p = h.H1_poincare;
    html += heresy('H1 Poincaré', 'norm=' + p.embedding_norm.toFixed(4) + ' · depth=' + p.depth + ' · freq=' + p.frequency +
      (p.summary ? '<br><span style="color:#9A9080">' + p.summary + '</span>' : ''));

    var t = h.H3_ternary;
    var tcells = t.ternary_vector.map(function(v) {
      return v > 0 ? '<div class="cell cell-pos">+1</div>' :
             v < 0 ? '<div class="cell cell-neg">−1</div>' :
             '<div class="cell cell-zero">0</div>';
    }).join('');
    html += heresy('H3 Ternary', t.nonzero_dims + '/10 active · ' + t.compression + '<div class="ternary">' + tcells + '</div>');

    var w = h.H4_wave;
    html += heresy('H4 Wave', 'dominant_freq=' + w.dominant_frequency + ' · energy=' + w.spectral_energy.toFixed(4) + '<br>' +
      '<span class="mono">mag: [' + w.magnitude_profile.map(function(m){return m.toFixed(3)}).join(', ') + ']</span>');

    var k = h.H7_kan;
    var klines = Object.entries(k.predicted_norms).map(function(pair) {
      return '<span class="mono">' + pair[0] + ' norm=' + pair[1].toFixed(4) + '</span>';
    }).join('<br>');
    html += heresy('H7 KAN', k.n_patterns + ' patterns<br>' + klines);

    var lq = h.H2_liquid;
    html += heresy('H2 Liquid', 'experts=' + JSON.stringify(lq.active_experts) + ' · complexity=' + lq.complexity.toFixed(4) + '<br>' +
      '<span class="mono">weights=[' + lq.expert_weights.map(function(x){return x.toFixed(4)}).join(', ') + ']</span>');

    var hv = h.H8_hive;
    var bars = '';
    var dcolors = {theology:'#C8A55C',nature:'#4CAF50',law:'#2196F3',narrative:'#E74C3C'};
    Object.entries(hv.domain_weights).forEach(function(pair) {
      var dn = pair[0], dw = pair[1];
      var pct = (dw * 100).toFixed(1);
      var color = dcolors[dn] || '#C8A55C';
      bars += '<div class="bar-row"><div class="bar-label">' + dn + '</div>' +
        '<div class="bar-track"><div class="bar-fill" style="width:' + pct + '%;background:' + color + '"></div></div>' +
        '<div class="bar-val">' + pct + '%</div></div>';
    });
    html += heresy('H8 Hive', 'domain: <b>' + hv.primary_domain + '</b>' + bars);

    var tt = h.H6_ttt;
    html += heresy('H6 TTT', 'delta=' + tt.adaptation_delta.toFixed(8) + ' · adapted=' + (tt.adapted ? 'YES' : 'NO'));

    var mm = h.H5_memory;
    var mrows = mm.unified_results.map(function(r) {
      return '<div class="mem-row"><span class="mem-name">' + r.name + '</span><span class="mem-score">' + r.score.toFixed(4) + '</span></div>';
    }).join('');
    html += heresy('H5 Memory', mrows);

    resultsDiv.innerHTML = html;
  } catch(err) {
    loadingDiv.style.display = 'none';
    resultsDiv.innerHTML = '<div id="empty-state">Error: ' + err.message + '</div>';
  }
}

function heresy(label, content) {
  return '<div class="heresy"><div class="label">' + label + '</div><div class="val">' + content + '</div></div>';
}

resize();
window.addEventListener('resize', resize);
draw();
"""

    return html_head + js_app + "</script>\n</body>\n</html>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8777)
