"""Self-contained replay of real held-out epochs with saved model predictions."""
import json
import numpy as np
import pandas as pd
from .config import output_path
from .data import load_prepared, select_reference, channel_indices


def build_demo(cfg):
    out = output_path(cfg)
    X, meta, names = load_prepared(cfg)
    seed = cfg["evaluation"]["seeds"][0]
    count = 8
    predictions = pd.read_csv(out / "predictions.csv")
    frame = predictions[(predictions.model == "eegnet") & (predictions.channels == count)
                        & (predictions.seed == seed) & (predictions.fold == 0)]
    # First three chronological trials per held-out participant, no accuracy-based selection.
    frame = frame.sort_values(["subject", "run", "onset"]).groupby("subject", sort=True).head(3)
    trial_index = {row.trial_id: i for i, row in meta.iterrows()}
    selected = select_reference(X, names, count)
    trials = []
    for row in frame.itertuples():
        idx = trial_index[row.trial_id]
        trials.append({"id": row.trial_id, "subject": int(row.subject), "run": int(row.run),
                       "truth": int(row.label), "prediction": int(row.prediction), "prob_right": float(row.prob_right),
                       "signals": np.round(selected[idx], 3).tolist()})
    if not trials:
        raise RuntimeError("No real held-out EEGNet predictions available for demo.")
    summary = pd.read_csv(out / "summary.csv")
    score = summary[(summary.model == "eegnet") & (summary.channels == count)].iloc[0]
    payload = {"trials": trials, "channels": [names[i] for i in channel_indices(names, count)],
               "sfreq": cfg["preprocessing"]["sfreq"], "duration": cfg["preprocessing"]["duration"],
               "balanced_accuracy": float(score.balanced_accuracy), "subjects": int(score.subjects),
               "seeds": cfg["evaluation"]["seeds"], "scope": cfg["scope"], "experiment": cfg["name"]}
    html = TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    destination = out / "demo" / "replay.html"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    print(destination, flush=True)
    return destination


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeuroEdge | Recorded EEG replay</title>
<style>
:root{font-family:Inter,system-ui,-apple-system,sans-serif;color:#122239;background:#f5f7fb;font-synthesis:none}
*{box-sizing:border-box}body{margin:0}button,select,input{font:inherit}button,select{border:1px solid #d6dfec;border-radius:9px;background:white;color:#122239;padding:10px 14px}button{cursor:pointer}button:focus-visible,select:focus-visible{outline:3px solid #82b4ff}button.primary{background:#235ed7;border-color:#235ed7;color:white}.wrap{max-width:1200px;margin:auto;padding:30px 28px 40px}header{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #dde4ef;padding-bottom:22px}.brand{font-size:25px;font-weight:800;letter-spacing:-1px}.brand span{color:#235ed7}.pill{font-size:11px;font-weight:750;letter-spacing:1px;border:1px solid #c6d9ed;color:#315377;border-radius:20px;padding:8px 12px}.eyebrow{font-size:11px;font-weight:750;letter-spacing:1.7px;color:#62748d;margin:26px 0 10px}h1{font-size:34px;letter-spacing:-1.1px;margin:0 0 10px;line-height:1.15}.intro{font-size:14px;line-height:1.6;max-width:820px;color:#60718a;margin:0}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:25px 0}.stat,.panel{background:white;border:1px solid #dde5ef;border-radius:13px}.stat{padding:17px 20px}.stat .value{font-size:28px;font-weight:750;letter-spacing:-.8px}.stat .label{font-size:12px;color:#60718a;margin-top:5px}.grid{display:grid;grid-template-columns:minmax(0,1fr) 292px;gap:18px}.panel{padding:21px}.panel-title{display:flex;justify-content:space-between;align-items:center;gap:14px;font-weight:750;font-size:15px}.muted{font-size:12px;color:#718198;font-weight:400;line-height:1.6}.signals{width:100%;height:330px;margin-top:15px;display:block}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:15px}.timeline{width:100%;accent-color:#235ed7;margin-top:13px}.clock{font-variant-numeric:tabular-nums;font-size:12px;color:#60718a}.prediction-title{font-size:11px;letter-spacing:1.3px;font-weight:750;color:#62748d}.decision{font-size:28px;letter-spacing:-.7px;font-weight:800;margin:16px 0 7px}.status{display:inline-block;background:#eef4ff;color:#235ed7;padding:6px 9px;border-radius:6px;font-size:11px}.prob-row{display:flex;justify-content:space-between;font-size:12px;margin-top:22px}.bar{height:7px;background:#e9eef6;border-radius:5px;margin-top:9px;overflow:hidden}.fill{height:100%;background:#235ed7;transition:width .2s}.divider{border-top:1px solid #e5eaf2;margin:25px 0 18px}.truth-toggle{display:flex;gap:7px;font-size:12px;line-height:1.5}.truth{margin-top:12px;font-size:13px;font-weight:650;min-height:19px}.small{font-size:11px;line-height:1.7;color:#718198}.note{margin-top:18px;padding:15px 19px;border:1px solid #dce5f1;border-radius:10px;color:#586e89;font-size:12px;line-height:1.65;background:#eef3fa}.footer{display:flex;justify-content:space-between;gap:15px;margin-top:23px;font-size:11px;color:#78889d}.tag{font-size:11px;color:#0b8279;background:#e8f7f4;border-radius:5px;padding:5px 8px}a{color:#235ed7}@media(max-width:800px){header{flex-direction:column;align-items:flex-start;gap:14px}.brand{white-space:nowrap}.grid{grid-template-columns:1fr}.stats{gap:8px}.stat{padding:14px}.stat .value{font-size:24px}.wrap{padding:22px 16px}h1{font-size:29px}.footer{flex-direction:column}.signals{height:300px}.pill{font-size:9px}}
</style></head><body><main class="wrap">
<header><div class="brand">Neuro<span>Edge</span><span style="font-size:11px;font-weight:500;color:#718198;letter-spacing:0;margin-left:12px">RESEARCH REPLAY</span></div><div class="pill">RECORDED DATA · OFFLINE STUDY</div></header>
<div class="eyebrow">MOTOR IMAGERY / SUBJECT-DISJOINT EVALUATION</div>
<h1>A closer look at imagined movement.</h1>
<p class="intro">Replay a real EEG epoch from an unseen participant. Explore eight sensorimotor channels and the model’s saved left-versus-right hand prediction.</p>
<section class="stats"><div class="stat"><div class="value" id="accuracy"></div><div class="label">Subject-mean balanced accuracy · 8 channels</div></div><div class="stat"><div class="value" id="cohort"></div><div class="label">Held-out participants across outer folds</div></div><div class="stat"><div class="value">8 / 64</div><div class="label">Available electrodes in this configuration</div></div></section>
<section class="grid"><div class="panel"><div class="panel-title"><span>Recorded EEG epoch</span><span class="tag">PhysioNet · EEGMMIDB</span></div><div class="muted" id="trial-meta"></div>
<canvas class="signals" id="signals" aria-label="Eight channels of recorded EEG in microvolts"></canvas>
<input class="timeline" id="timeline" aria-label="Recorded epoch position" type="range" min="0" max="3" step="0.01" value="3">
<div class="controls"><button class="primary" id="play">Replay epoch</button><button id="previous" aria-label="Previous trial">←</button><button id="next" aria-label="Next trial">→</button><select id="trial-select" aria-label="Choose held-out EEG trial"></select><span class="clock" id="clock"></span></div>
</div><aside class="panel"><div class="prediction-title">SAVED EEGNET PREDICTION</div><div class="decision" id="decision"></div><div class="status" id="status"></div>
<div class="prob-row"><span>Left hand</span><strong id="p-left"></strong></div><div class="bar"><div class="fill" id="left-fill"></div></div>
<div class="prob-row"><span>Right hand</span><strong id="p-right"></strong></div><div class="bar"><div class="fill" id="right-fill"></div></div>
<p class="small">Softmax scores are model outputs, not calibrated probabilities of correctness.</p><div class="divider"></div>
<label class="truth-toggle"><input id="reveal" type="checkbox">Reveal recorded task label</label><div class="truth" id="truth"></div>
<p class="small">This participant was excluded from training, normalization fitting, and validation for this model. Predictions appear after the complete 3-second epoch.</p>
</aside></section>
<div class="note"><strong>What you are seeing:</strong> real, preprocessed EEG and predictions computed earlier by the Python pipeline. This page replays saved results; it does not run live inference. Filtering is offline and non-causal. No EEG device is connected, and no clinical claim is made.</div>
<footer class="footer"><span id="scope"></span><span>Source: <a href="https://physionet.org/content/eegmmidb/1.0.0/" target="_blank" rel="noopener">Schalk / PhysioNet</a> · ODC Attribution License</span></footer>
</main><script id="dataset" type="application/json">__PAYLOAD__</script><script>
'use strict';const D=JSON.parse(document.getElementById('dataset').textContent);let selected=0,position=3,running=false,last=0;
const $=id=>document.getElementById(id);$('accuracy').textContent=(100*D.balanced_accuracy).toFixed(1)+'%';$('cohort').textContent=D.subjects;$('scope').textContent=D.experiment+' · '+D.seeds.length+' training seed(s) · fixed cohort';
D.trials.forEach((t,i)=>{const o=document.createElement('option');o.value=i;o.textContent=t.id;$('trial-select').append(o)});
function render(){const t=D.trials[selected],complete=position>=D.duration;const cls=t.prediction?'Right hand':'Left hand';$('trial-meta').textContent='Participant S'+String(t.subject).padStart(3,'0')+' · run '+t.run+' · 1–4 s after imagery cue · µV, auto-scaled';$('clock').textContent=position.toFixed(2)+' / '+D.duration.toFixed(2)+' s';$('timeline').value=position;$('decision').textContent=complete?cls:'Acquiring epoch…';$('status').textContent=complete?'Complete window · stored output':'Replay in progress';$('p-left').textContent=complete?((1-t.prob_right)*100).toFixed(1)+'%':'—';$('p-right').textContent=complete?(t.prob_right*100).toFixed(1)+'%':'—';$('left-fill').style.width=complete?((1-t.prob_right)*100)+'%':'0';$('right-fill').style.width=complete?(t.prob_right*100)+'%':'0';$('truth').textContent=$('reveal').checked&&complete?('Task: '+(t.truth?'right':'left')+' hand · '+(t.truth===t.prediction?'correct':'incorrect')):'';draw(t)}
function draw(t){const c=$('signals'),r=c.getBoundingClientRect(),dpr=window.devicePixelRatio||1;c.width=Math.round(r.width*dpr);c.height=Math.round(r.height*dpr);const ctx=c.getContext('2d');ctx.scale(dpr,dpr);const w=r.width,h=r.height,left=39,right=10,top=12,bottom=23,row=(h-top-bottom)/8;ctx.clearRect(0,0,w,h);ctx.font='10px system-ui';ctx.strokeStyle='#eef2f7';ctx.fillStyle='#8795a8';for(let sec=0;sec<=3;sec++){const x=left+(w-left-right)*sec/3;ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,h-bottom);ctx.stroke();ctx.fillText(sec+'s',x-6,h-5)}const max=Math.max(10,...t.signals.flat().map(Math.abs));for(let ch=0;ch<8;ch++){const y=top+row*(ch+.5);ctx.fillStyle='#60718a';ctx.fillText(D.channels[ch],0,y+3);ctx.strokeStyle='#f1f4f8';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(w-right,y);ctx.stroke();ctx.strokeStyle='#306bd6';ctx.lineWidth=1.1;ctx.beginPath();const stop=Math.min(t.signals[ch].length,Math.round(position*D.sfreq));for(let j=0;j<stop;j++){const x=left+(w-left-right)*j/(t.signals[ch].length-1),v=y-t.signals[ch][j]/max*row*.44;if(j===0)ctx.moveTo(x,v);else ctx.lineTo(x,v)}ctx.stroke()}const x=left+(w-left-right)*position/3;ctx.strokeStyle='#0d9488';ctx.lineWidth=1.3;ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,h-bottom);ctx.stroke()}
function choose(i){selected=(i+D.trials.length)%D.trials.length;position=3;running=false;$('play').textContent='Replay epoch';$('trial-select').value=selected;render()}
$('play').onclick=()=>{if(running){running=false;$('play').textContent='Continue replay'}else{if(position>=3)position=0;running=true;last=performance.now();$('play').textContent='Pause';render();requestAnimationFrame(tick)}};
function tick(now){if(!running)return;position=Math.min(3,position+(now-last)/1000);last=now;render();if(position>=3){running=false;$('play').textContent='Replay epoch'}else requestAnimationFrame(tick)}
$('previous').onclick=()=>choose(selected-1);$('next').onclick=()=>choose(selected+1);$('trial-select').onchange=e=>choose(Number(e.target.value));$('timeline').oninput=e=>{running=false;position=Number(e.target.value);$('play').textContent=position>=3?'Replay epoch':'Continue replay';render()};$('reveal').onchange=render;window.addEventListener('resize',render);render();
</script></body></html>'''

