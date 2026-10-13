"""FAIE LIVE — real-time inference dashboard with honest token counting.

Streams real generation from GLM-5.2-REAM (380.93B) over SSE and shows tok/s NEXT TO
expert residency, always. A tok/s number without its residency is meaningless here:
below 100% residency the engine silently skips non-resident experts and runs FASTER
while computing LESS. The dashboard makes that impossible to hide.

  python faie_live.py          -> http://127.0.0.1:8790
"""
import sys, os, json, time, ctypes, threading, queue
sys.path.insert(0, r"E:\ABU\NEOCORTEX")
import torch
from gguf import GGUFReader
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

from faie_kernel import load_kernel
from faie_generate import FAIEInference
from expert_locality import GGUFTokenizer
import faie_generate

MODEL_PATH = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"
FAIE_DIR = r"E:\ABU\NEOCORTEX\faie_model"
VP = ctypes.c_void_p
PORT = 8790

STATE = {"ready": False, "status": "booting", "engine": None, "tok": None, "lib": None}
LOCK = threading.Lock()


def boot():
    STATE["status"] = "reading GGUF (129.4 GB memmap)"
    reader = GGUFReader(MODEL_PATH, "r")
    lib = load_kernel()
    lib.get_experts_requested.restype = ctypes.c_longlong
    lib.get_experts_executed.restype = ctypes.c_longlong
    STATE["status"] = "building tokenizer"
    tok = GGUFTokenizer(reader)
    STATE["status"] = "uploading weights to GPU"
    engine = FAIEInference(reader, lib, FAIE_DIR, n_layers=78)
    faie_generate.FAIEInference._moe_ffn_warm = lambda self, x, li: lib.full_moe_layer_v2(
        VP(x.data_ptr()), li, VP(self._norm_buf.data_ptr()),
        VP(self._gate_buf.data_ptr()), VP(self._mega_temp.data_ptr()))
    STATE["status"] = "cold run + capturing CUDA graphs"
    engine.forward_single_token(1)
    lib.sync_device(); torch.cuda.synchronize()
    STATE.update(engine=engine, tok=tok, lib=lib, ready=True, status="ready")
    print(f"\n  FAIE LIVE ready on http://127.0.0.1:{PORT}\n")


def gpu_stats():
    out = []
    for i in range(torch.cuda.device_count()):
        free_b, total_b = torch.cuda.mem_get_info(i)
        out.append({"idx": i, "used_gb": round((total_b - free_b) / 1024**3, 2),
                    "total_gb": round(total_b / 1024**3, 1)})
    return out


class Req(BaseModel):
    prompt: str = "الرحمن الرحيم"
    max_tokens: int = 64
    mode: str = "free"          # "free" | "locked"


app = FastAPI()


@app.get("/api/status")
def status():
    e = STATE["engine"]
    return {"ready": STATE["ready"], "status": STATE["status"],
            "resident": len(e._expert_cache) if e else 0,
            "cap": e._expert_cache_max if e else 0,
            "total_experts": 9600, "gpus": gpu_stats()}


@app.post("/api/generate")
def generate(req: Req):
    if not STATE["ready"]:
        return {"error": "not ready"}

    def stream():
        with LOCK:
            engine, tok, lib = STATE["engine"], STATE["tok"], STATE["lib"]
            ids = tok.encode(req.prompt, limit=64) or [1]
            # LOCKED mode re-feeds one token so every top-8 stays resident -> the
            # honest full-fidelity ceiling. FREE mode is true generation.
            locked = req.mode == "locked"

            for t in ids[:-1]:                     # prefill
                engine.forward_single_token(t)
            lib.sync_device(); torch.cuda.synchronize()

            if locked:
                # Locked mode must GUARANTEE 100% residency, not merely report it.
                # Repeatedly run the token and page in whatever the router asked for
                # but could not find, until the miss log comes back empty.
                _prep0 = json.dumps({"ev": "prep", "msg": "paging in the working set for this prompt..."})
                yield "data: " + _prep0 + "\n\n"
                for rnd in range(12):
                    engine.forward_single_token(ids[-1])
                    lib.sync_device(); torch.cuda.synchronize()
                    paged, _ = engine.page_in_misses()
                    yield "data: " + json.dumps({
                        "ev": "prep", "msg": f"round {rnd+1}: paged {paged} experts, "
                                             f"{len(engine._expert_cache):,} resident"}) + "\n\n"
                    if paged == 0:
                        break

            for _ in range(3):                     # warm clocks
                engine.forward_single_token(ids[-1])
            lib.sync_device(); torch.cuda.synchronize()

            lib.reset_expert_counters()
            cur, n, t_start, lat = ids[-1], 0, time.perf_counter(), []
            yield f"data: {json.dumps({'ev':'start','prompt':req.prompt,'prefill':len(ids)})}\n\n"

            for step in range(req.max_tokens):
                s = time.perf_counter()
                logits = engine.forward_single_token(ids[-1] if locked else cur)
                lib.sync_device(); torch.cuda.synchronize()
                dt = (time.perf_counter() - s) * 1000
                lat.append(dt)
                cur = int(torch.argmax(logits.squeeze().float()).item())
                n += 1
                req_c, exe_c = lib.get_experts_requested(), lib.get_experts_executed()
                resid = 100.0 * exe_c / req_c if req_c else 0.0
                elapsed = time.perf_counter() - t_start
                yield "data: " + json.dumps({
                    "ev": "tok", "i": n, "text": tok.decode([cur]),
                    "ms": round(dt, 2),
                    "inst_tps": round(1000.0 / dt, 2),
                    "avg_tps": round(n / elapsed, 2),
                    "residency": round(resid, 2),
                    "experts_exec": exe_c, "experts_req": req_c,
                    "resident": len(engine._expert_cache),
                }) + "\n\n"

            elapsed = time.perf_counter() - t_start
            req_c, exe_c = lib.get_experts_requested(), lib.get_experts_executed()
            resid = 100.0 * exe_c / req_c if req_c else 0.0
            lat_s = sorted(lat)
            yield "data: " + json.dumps({
                "ev": "done", "tokens": n, "seconds": round(elapsed, 3),
                "avg_tps": round(n / elapsed, 2),
                "p50_ms": round(lat_s[len(lat_s)//2], 2),
                "best_ms": round(lat_s[0], 2),
                "residency": round(resid, 2),
                "honest": resid >= 99.99,
                "mode": req.mode,
            }) + "\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>FAIE LIVE — GLM-5.2-REAM on one RTX 5090</title>
<style>
:root{--bg:#0A0E12;--panel:#111820;--rule:#22303B;--ink:#DCE7EE;--ink2:#93A6B2;
--ink3:#63757F;--ok:#22E39B;--warn:#FF7A4D;--mono:ui-monospace,"Cascadia Mono",Consolas,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--mono);font-size:14px;
font-variant-numeric:tabular-nums;line-height:1.6}
.wrap{max-width:1120px;margin:0 auto;padding:28px 20px 80px}
.eyebrow{font-size:11px;letter-spacing:.2em;text-transform:uppercase;color:var(--ink3)}
h1{font-size:26px;font-weight:700;letter-spacing:-.03em;margin:12px 0 4px}
.sub{color:var(--ink2);font-size:13px;margin-bottom:24px}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));margin-bottom:18px}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:4px;padding:15px 17px}
.lab{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink3)}
.val{font-size:30px;font-weight:700;letter-spacing:-.04em;margin-top:5px;line-height:1}
.val small{font-size:12px;font-weight:500;color:var(--ink2);margin-left:5px;letter-spacing:0}
.ok{color:var(--ok)} .warn{color:var(--warn)}
.hint{font-size:11px;color:var(--ink3);margin-top:6px}
.ctl{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:16px}
input,select,button{font-family:var(--mono);font-size:13px;background:var(--panel);
color:var(--ink);border:1px solid var(--rule);border-radius:3px;padding:9px 12px}
input{flex:1;min-width:260px}
button{background:var(--ok);color:#04140F;border-color:var(--ok);font-weight:700;cursor:pointer;padding:9px 20px}
button:disabled{opacity:.4;cursor:not-allowed}
#spark{width:100%;height:60px;display:block}
.out{background:var(--panel);border:1px solid var(--rule);border-radius:4px;padding:18px;
min-height:150px;white-space:pre-wrap;word-break:break-word;font-size:15px;line-height:1.85;
direction:auto;unicode-bidi:plaintext}
.cursor{display:inline-block;width:8px;height:17px;background:var(--ok);vertical-align:-3px;
animation:b 1s steps(2) infinite}@keyframes b{0%,50%{opacity:1}51%,100%{opacity:0}}
.banner{border-radius:4px;padding:12px 16px;margin-bottom:16px;font-size:12.5px;border:1px solid}
.b-ok{border-color:var(--ok);background:#0C2A22;color:var(--ok)}
.b-warn{border-color:var(--warn);background:#2A140C;color:var(--warn)}
h2{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--ink3);
border-bottom:1px solid var(--rule);padding-bottom:8px;margin:30px 0 14px}
</style></head><body><div class="wrap">
<div class="eyebrow">FAIE · Fused Arabic Inference Engine · NEOCORTEX</div>
<h1>GLM-5.2-REAM — 380.93B params, one RTX 5090</h1>
<div class="sub" id="boot">booting…</div>

<div class="ctl">
  <input id="p" value="الرحمن الرحيم الرحمة" spellcheck="false">
  <select id="mode">
    <option value="locked">Locked — 100% resident (true ceiling)</option>
    <option value="free">Free generation (shows cache misses)</option>
  </select>
  <select id="n"><option>32</option><option selected>64</option><option>128</option></select>
  <button id="go" disabled>RUN</button>
</div>

<div id="banner" class="banner b-warn" style="display:none"></div>

<div class="grid">
  <div class="card"><div class="lab">Tokens / sec</div><div class="val ok" id="tps">—</div>
    <div class="hint" id="inst">instant —</div></div>
  <div class="card"><div class="lab">Expert residency</div><div class="val" id="res">—</div>
    <div class="hint" id="resd">— of — top-8 picks executed</div></div>
  <div class="card"><div class="lab">Tokens emitted</div><div class="val" id="cnt">0</div>
    <div class="hint" id="el">0.00 s elapsed</div></div>
  <div class="card"><div class="lab">Latency / token</div><div class="val" id="ms">—</div>
    <div class="hint" id="best">best —</div></div>
  <div class="card"><div class="lab">Experts in VRAM</div><div class="val" id="cache">—</div>
    <div class="hint">of 9,600 total</div></div>
</div>

<canvas id="spark"></canvas>
<h2>Live output — throughput probe, NOT coherent text</h2>
<div class="banner b-warn" style="display:block">
<b>READ THIS BEFORE JUDGING THE TEXT.</b> This engine has <b>no KV cache and no positional
encoding yet</b> (open task #61). Every token is a full 78-layer forward pass computed in
complete isolation — the model has no memory of the tokens before it. The arithmetic is
correct and the speed is real, but the output will be repetitive and meaningless until
attention-over-history lands. What this page proves is <b>throughput</b>, not fluency.
</div>
<div class="out" id="out"><span class="cursor"></span></div>
<h2>Why residency is shown next to tok/s</h2>
<div class="sub">Below 100% residency the engine skips experts that are not in VRAM — it runs
<b>faster</b> while computing <b>less</b>. A tok/s figure without its residency is not a
measurement. Locked mode pages in the prompt's working set until the miss log returns
empty, so its number is measured at a verified 100%.</div>
</div>
<script>
const $=i=>document.getElementById(i), lat=[];
let boot=setInterval(async()=>{
  const r=await (await fetch('/api/status')).json();
  $('boot').textContent = r.ready
    ? `ready · ${r.resident.toLocaleString()} / 9,600 experts resident (cap ${r.cap.toLocaleString()}) · `
      + r.gpus.map(g=>`GPU${g.idx} ${g.used_gb}/${g.total_gb} GB`).join(' · ')
    : 'booting — '+r.status+' …';
  if(r.ready){ $('go').disabled=false; $('cache').textContent=r.resident.toLocaleString();
    clearInterval(boot); }
},1200);

function draw(){
  const c=$('spark'), x=c.getContext('2d'), W=c.width=c.clientWidth*2, H=c.height=120;
  x.clearRect(0,0,W,H); if(lat.length<2) return;
  const mx=Math.max(...lat)*1.15, n=lat.length;
  x.strokeStyle='#22303B'; x.lineWidth=2;
  [0.5,1].forEach(f=>{x.beginPath();x.moveTo(0,H-H*f*0.9);x.lineTo(W,H-H*f*0.9);x.stroke();});
  // 33.33ms = the 30 tok/s line
  const yT=H-(33.33/mx)*H*0.9;
  x.strokeStyle='#FF7A4D'; x.setLineDash([8,6]); x.beginPath();
  x.moveTo(0,yT); x.lineTo(W,yT); x.stroke(); x.setLineDash([]);
  x.strokeStyle='#22E39B'; x.lineWidth=3; x.beginPath();
  lat.forEach((v,i)=>{const px=i/(n-1)*W, py=H-(v/mx)*H*0.9; i?x.lineTo(px,py):x.moveTo(px,py);});
  x.stroke();
  x.fillStyle='#63757F'; x.font='20px ui-monospace';
  x.fillText('30 tok/s (33.3 ms)', 8, yT-9);
}

$('go').onclick=async()=>{
  $('go').disabled=true; lat.length=0; $('out').innerHTML='<span class="cursor"></span>';
  $('banner').style.display='none'; let txt='';
  const res=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({prompt:$('p').value,max_tokens:+$('n').value,mode:$('mode').value})});
  const rd=res.body.getReader(), dec=new TextDecoder(); let buf='';
  while(true){ const {done,value}=await rd.read(); if(done) break;
    buf+=dec.decode(value,{stream:true});
    let p; while((p=buf.indexOf('\n\n'))>=0){
      const line=buf.slice(0,p); buf=buf.slice(p+2);
      if(!line.startsWith('data: ')) continue;
      const d=JSON.parse(line.slice(6));
      if(d.ev==='prep'){
        const b=$('banner'); b.style.display='block'; b.className='banner b-warn';
        b.innerHTML='<b>PREPARING</b> — '+d.msg;
      } else if(d.ev==='tok'){
        txt+=d.text; lat.push(d.ms);
        $('out').innerHTML=txt.replace(/</g,'&lt;')+'<span class="cursor"></span>';
        $('tps').textContent=d.avg_tps; $('inst').textContent='instant '+d.inst_tps+' tok/s';
        $('cnt').textContent=d.i; $('ms').textContent=d.ms+' ms';
        $('el').textContent=(d.i/d.avg_tps).toFixed(2)+' s elapsed';
        $('best').textContent='best '+Math.min(...lat).toFixed(2)+' ms';
        const full=d.residency>=99.99;
        $('res').textContent=d.residency.toFixed(1)+'%';
        $('res').className='val '+(full?'ok':'warn');
        $('resd').textContent=d.experts_exec.toLocaleString()+' of '+d.experts_req.toLocaleString()+' top-8 picks executed';
        $('cache').textContent=d.resident.toLocaleString();
        $('tps').className='val '+(full?'ok':'warn');
        draw();
      } else if(d.ev==='done'){
        const b=$('banner'); b.style.display='block';
        b.className='banner '+(d.honest?'b-ok':'b-warn');
        b.innerHTML = d.honest
          ? `<b>HONEST RESULT</b> — ${d.tokens} tokens in ${d.seconds}s = <b>${d.avg_tps} tok/s</b> at 100% expert residency. p50 ${d.p50_ms} ms, best ${d.best_ms} ms. Every top-8 expert executed.`
          : `<b>NOT A VALID SPEED CLAIM</b> — ${d.avg_tps} tok/s was measured at only ${d.residency}% residency. Non-resident experts were skipped, so the model did LESS work than specified. Use Locked mode for the real ceiling.`;
        $('go').disabled=false;
      }
    }}
};
addEventListener('resize',draw);
</script></body></html>"""


if __name__ == "__main__":
    threading.Thread(target=boot, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
