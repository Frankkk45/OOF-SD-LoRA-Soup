"""A small local web page for trying the models and reading the test results (Python standard library only).

  python tools/app.py                         # then open http://127.0.0.1:7860
  python tools/app.py --device cpu            # no GPU (slower: a few seconds per text)
  python tools/app.py --no-models             # only the page and the results tables (no torch needed)

Qwen3.5-4B is loaded once, on the first request that needs it; each LoRA adapter is loaded on first use and then
switched with set_adapter. Thesis 1 models are loaded on first use. All result numbers on the page are read from
reference/results.json, reference/thesis1_results.json, reference/wrime_edom18_protocol.json and
reference/published_results.json.
"""
import argparse
import html
import importlib.util
import json
import threading
import time
from decimal import ROUND_HALF_UP, Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from common import DATASETS, DEVICE_CHOICES, ROOT, domain, model_path, pick_device, read_json

QWEN_MODELS = {
    'soup': ('Final model (SD uniform-6 LoRA soup)',
             'Qwen3.5-4B with one rank-48 LoRA adapter: the exact weight-space average of six out-of-fold '
             'self-distillation models.'),
    'A_s42': ('Direct LoRA fine-tuning (A, seed 42)',
              'Qwen3.5-4B with one rank-8 LoRA adapter trained on the gold labels only.'),
}
DEMO_MODELS = {'usual': ['soup', 'A_s42', 'thesis1'], 'virus': ['soup', 'A_s42'], 'goemotions': ['soup'], 'wrime': ['soup']}
LANG = {'usual': 'zh', 'virus': 'zh', 'goemotions': 'en', 'wrime': 'ja'}
# Short invented inputs, one per line of the example row.
EXAMPLES = {
    'usual': ['终于拿到offer了，激动得睡不着！', '排了三个小时的队，结果系统坏了，气死我了。', '外婆走了，再也吃不到她做的饭了。'],
    'virus': ['小区又封了，什么时候才是个头啊。', '听说隔壁楼有确诊病例，晚上都不敢出门了。', '医护人员辛苦了，谢谢你们一直坚守！'],
    'goemotions': ['This is the best news I have heard all week!', "They cancelled the show again. Unbelievable.",
                   'Wait, he actually said that out loud? No way.'],
    'wrime': ['明日から旅行だ、楽しみすぎて眠れない！', 'また電車が遅れて会議に間に合わなかった。最悪。', '夜中に変な物音がして怖い…'],
}


def load_thesis1():
    """thesis1/models.py as a module (it imports torch only when a model is loaded), or None."""
    path = ROOT / 'thesis1' / 'models.py'
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location('thesis1_models', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def demo_models(t1):
    """{dataset: [dict(id, name, description, available)]} of the models offered on the page."""
    out = {}
    for ds in DATASETS:
        rows = []
        for m in DEMO_MODELS[ds]:
            if m == 'thesis1':
                for info in (t1.THESIS1_MODELS if t1 else []):
                    ok = (ROOT / 'weights' / 'thesis1' / info['file']).is_file() and \
                         (ROOT / 'models' / 'chinese-roberta-wwm-ext' / 'config.json').is_file()
                    rows.append(dict(id='t1:' + info['key'], name='Thesis 1: ' + info['display_name'],
                                     description=info['description'], available=ok))
                continue
            try:
                model_path(ds, m)
                ok = True
            except SystemExit:
                ok = False
            rows.append(dict(id=m, name=QWEN_MODELS[m][0], description=QWEN_MODELS[m][1], available=ok))
        out[ds] = rows
    return out


class Engine:
    """Loads models on first use; one request at a time."""

    def __init__(self, args, t1):
        self.args, self.t1_module, self.lock = args, t1, threading.Lock()
        self.scorer, self.t1 = None, {}

    def t1_device(self):
        return pick_device(self.args.device)

    def predict(self, ds, model, text):
        with self.lock:
            if model.startswith('t1:'):
                key = model[3:]
                if key not in self.t1:
                    self.t1[key] = self.t1_module.load_model(key, device=self.t1_device(), root=ROOT)
                return self.t1[key].predict_proba([text])[0]
            if self.scorer is None:
                from scorer import Scorer
                self.scorer = Scorer(self.args.model_dir, device=self.args.device, threads=self.args.threads)
            self.scorer.use(model_path(ds, model), name=f'{ds}:{model}')
            return self.scorer.score([text], ds)[0]


# ------------------------------------------------------------------ results tables
def esc(s):
    return html.escape(str(s), quote=True)


def fmt(v):
    """Whole numbers as reported; otherwise two decimals, half up (72.255 -> 72.26, not 72.25 from binary rounding)."""
    if v is None:
        return '&ndash;'
    if isinstance(v, int):
        return str(v)
    return str(Decimal(repr(round(v, 6))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


def test_scores(results, dom, models):
    """Test accuracy and Macro-F1 in percent, averaged over `models`; None if any is missing."""
    acc, mf1 = [], []
    for m in models:
        r = ((results or {}).get(dom) or {}).get(m, {}).get('test')
        if not isinstance(r, dict):
            return None
        acc.append(100 * r['accuracy'])
        mf1.append(100 * r['macro_f1'])
    return sum(acc) / len(acc), sum(mf1) / len(mf1)


def table(columns, rows, numeric):
    """columns: header names; rows: dict(cells, kind, final); numeric: indices of the columns to compare."""
    best = {}
    for c in numeric:
        values = [r['cells'][c] for r in rows if isinstance(r['cells'][c], (int, float))]
        best[c] = max(values) if values else None
    head = ''.join(f'<th{" class=num" if i in numeric else ""}>{esc(c)}</th>' for i, c in enumerate(columns))
    body = []
    for r in rows:
        cells = []
        for i, v in enumerate(r['cells']):
            if i in numeric:
                text = fmt(v)
                if isinstance(v, (int, float)) and best[i] is not None and abs(v - best[i]) < 1e-9:
                    text = f'<strong>{text}</strong>'
                cells.append(f'<td class=num>{text}</td>')
            elif i == 1:
                cells.append(f'<td><span class="tag {r["kind"]}">{esc(v)}</span></td>')
            else:
                cells.append(f'<td>{v}</td>')
        body.append(f'<tr{" class=final" if r.get("final") else ""}>' + ''.join(cells) + '</tr>')
    return f'<div class=scroll><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def published_cell(p):
    cite = ', '.join(str(x) for x in (p.get('authors'), p.get('year'), p.get('venue')) if x)
    name = f'<a href="{esc(p["url"])}" target=_blank rel=noopener>{esc(p["system"])}</a>' if p.get('url') else esc(p['system'])
    return f'{name}<span class=cite>{esc(cite)}</span>'


def published_rows(published, ds, metrics=('accuracy', 'macro_f1')):
    entries = (((published or {}).get('datasets') or {}).get(ds) or {}).get('results') or []
    return [dict(cells=[published_cell(p), 'Published'] + [p.get(m) for m in metrics] + [esc(p.get('short_note', ''))],
                 kind='pub') for p in entries]


OURS_ROWS = [
    (['A_s42'], 'Direct LoRA fine-tuning (A, seed 42)', 'one rank-8 LoRA adapter'),
    (['A_s42', 'A_s43', 'A_s44'], 'Direct LoRA fine-tuning (A), mean of 3 seeds', 'seeds 42, 43, 44'),
    (['a_soup'], 'A uniform-3 LoRA soup', 'exact weight average of the 3 A seeds'),
    (['SD_s42', 'SD_s43', 'SD_s44'], 'Out-of-fold self-distillation (SD), mean of 3 seeds', 'seeds 42, 43, 44'),
    (['soup'], 'Final model (SD uniform-6 LoRA soup)', 'one rank-48 adapter, one forward pass'),
]


def our_rows(results, ds):
    rows = []
    for models, name, note in OURS_ROWS:
        s = test_scores(results, ds, models)
        if s:
            rows.append(dict(cells=[esc(name), 'This work', s[0], s[1], esc(note)], kind='ours', final=models == ['soup']))
    return rows


def thesis1_rows(t1_results, t1):
    rows = []
    for info in (t1.THESIS1_MODELS if t1 else []):
        r = (t1_results or {}).get(info['key'], {}).get('test')
        if isinstance(r, dict):
            note = 'chinese-roberta-wwm-ext, same splits' + ('; full Thesis 1 model' if info['key'].endswith('_topo') else '')
            rows.append(dict(cells=[esc(info['display_name']), 'Thesis 1', 100 * r['accuracy'], 100 * r['macro_f1'], esc(note)],
                             kind='t1'))
    return rows


def official_table(ref):
    published, results = ref['published'] or {}, ref['results']
    rows, seen = [], set()
    for p in (published.get('official_two_domain_score') or {}).get('results', []):
        seen.add(p.get('id'))
        rows.append(dict(cells=[esc(p['system']), 'Published', p.get('usual_macro_f1'), p.get('virus_macro_f1'), p.get('score'),
                                esc(p.get('short_note', ''))], kind='pub'))
    by_id = {}
    for ds in ('usual', 'virus'):
        for p in ((published.get('datasets') or {}).get(ds) or {}).get('results', []):
            by_id.setdefault(p['id'], {})[ds] = p
    for pid, both in by_id.items():
        if pid not in seen and len(both) == 2 and both['usual'].get('macro_f1') and both['virus'].get('macro_f1'):
            u, v = both['usual']['macro_f1'], both['virus']['macro_f1']
            rows.append(dict(cells=[esc(both['usual']['system']), 'Published', u, v, (u + v) / 2,
                                    esc('score computed from the two reported values')], kind='pub'))
    for models, name, note in ((['A_s42'], 'Direct LoRA fine-tuning (A, seed 42)', 'one model per domain'),
                               (['soup'], 'Final models (SD uniform-6 LoRA soups)', 'one model per domain')):
        u, v = test_scores(results, 'usual', models), test_scores(results, 'virus', models)
        if u and v:
            rows.append(dict(cells=[esc(name), 'This work', u[1], v[1], (u[1] + v[1]) / 2, esc(note)], kind='ours',
                             final=models == ['soup']))
    return table(['System', 'Type', 'General Macro-F1', 'COVID-19 Macro-F1', 'Official score', 'Note'], rows, [2, 3, 4])


def edom18_table(ref):
    rows = published_rows(ref['published'], 'wrime_edom18', ('macro_f1', 'weighted_f1'))
    models = (ref['edom18'] or {}).get('models') or {}
    for key, name, note in (('A_s42', 'Direct LoRA fine-tuning (A, seed 42)', 'trained under the 9-class protocol of this work'),
                            ('soup', 'Final model (SD uniform-6 LoRA soup)', 'trained under the 9-class protocol of this work')):
        r = models.get(key)
        if isinstance(r, dict) and 'macro_f1' in r:
            rows.append(dict(cells=[esc(name), 'This work', 100 * r['macro_f1'], 100 * r['weighted_f1'], esc(note)], kind='ours',
                             final=key == 'soup'))
    return table(['System', 'Type', 'Macro-F1', 'Weighted-F1', 'Note'], rows, [2, 3])


def results_section(ref, t1):
    parts = []
    datasets = (ref['published'] or {}).get('datasets') or {}
    for ds in DATASETS:
        rows = published_rows(ref['published'], ds)
        if ds == 'usual':
            rows += thesis1_rows(ref['thesis1'], t1)
        rows += our_rows(ref['results'], ds)
        info = datasets.get(ds) or {}
        protocol = f'<p class=proto>{esc(info["protocol"])}</p>' if info.get('protocol') else ''
        parts.append(f'<h3>{esc(domain(ds).title)}</h3>{protocol}'
                     + table(['System', 'Type', 'Accuracy', 'Macro-F1', 'Note'], rows, [2, 3]))
        if ds == 'virus':
            parts.append('<h3>Official SMP2020-EWECT score (mean of the general and COVID-19 Macro-F1)</h3>' + official_table(ref))
        if ds == 'wrime':
            if info.get('note'):
                parts.append(f'<p class=proto>{esc(info["note"])}</p>')
            alt = datasets.get('wrime_edom18') or {}
            parts.append('<h3>WRIME ver2 test split under the edom18 9-class relabelling</h3>'
                         + (f'<p class=proto>{esc(alt["protocol"])}</p>' if alt.get('protocol') else '') + edom18_table(ref))
    return ''.join(parts)


FOOTNOTES = [
    'All numbers are test-set scores in percent; the best value of each column is in bold. Rows of this work and of '
    'Thesis 1 are computed from the per-row predictions in reference/predictions.',
    'Published numbers are as reported by their authors and were not re-run here; protocol differences are given in '
    'reference/published_results.json. &ndash; means not reported.',
    'The 2020 winner is a large ensemble of several hundred models trained with extra data; the evaluation ranked teams '
    "by the official score, and the winner's accuracy comes from the final leaderboard.",
    'WRIME has no published result under the 9-class protocol used here, hence the second WRIME table: the WRIME models '
    'of this work, not retrained, scored on the official test split relabelled with the rule of the closest public '
    '9-class result.',
]

PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Emotion Classification Demo</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--muted:#667085;--line:#e3e6eb;--accent:#2f6fde;--bar:#dde6f7;--top:#2f6fde;
--pub:#eef1f5;--pubink:#475467;--t1:#f4efe6;--t1ink:#7a5b1e;--ours:#e6f0ff;--oursink:#1f4fae;--final:#f3f7ff}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--card:#1d2128;--ink:#e8eaee;--muted:#9aa3b2;--line:#2c323c;--accent:#6e9cf0;
--bar:#2a3550;--top:#6e9cf0;--pub:#262b34;--pubink:#b8c0cc;--t1:#2f2a21;--t1ink:#e0c58f;--ours:#1f2c45;--oursink:#a9c4f5;--final:#1a2233}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI","Hiragino Sans","Microsoft YaHei",sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px}h1{font-size:22px;margin:0 0 4px}h2{font-size:19px;margin:30px 0 6px}
h3{font-size:15px;margin:22px 0 4px}.sub{color:var(--muted);margin:0 0 18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:14px}
label{font-size:13px;color:var(--muted);display:block;margin-bottom:4px}
textarea{width:100%;min-height:100px;resize:vertical;padding:10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);
color:var(--ink);font:inherit}.row{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:10px}.row>div{flex:1 1 260px}
select{width:100%;padding:8px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink);font:inherit}
button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:9px 18px;font:inherit;cursor:pointer;margin-top:12px}
button:disabled{opacity:.55;cursor:not-allowed}.ex{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
.ex span{border:1px solid var(--line);border-radius:999px;padding:2px 10px;font-size:13px;cursor:pointer;color:var(--muted)}
.ex span:hover{color:var(--ink);border-color:var(--accent)}.desc{font-size:13px;color:var(--muted);margin-top:4px}
.result h2{font-size:20px;margin:0 0 10px}.result h2 small{font-size:13px;color:var(--muted);font-weight:400;margin-left:6px}
.bar{display:grid;grid-template-columns:150px 1fr 56px;align-items:center;gap:8px;margin:5px 0}
.bar small{color:var(--muted);margin-left:6px}.track{background:var(--bar);border-radius:6px;height:14px;overflow:hidden}
.fill{background:var(--muted);height:100%}.best .fill{background:var(--top)}.best{font-weight:600}
.meta{color:var(--muted);font-size:13px;margin-top:8px}.err{color:#d14343}.note{background:var(--pub);border-radius:8px;padding:8px 12px;
font-size:13px;color:var(--pubink)}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;background:var(--card)}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}th{font-size:12px;color:var(--muted);font-weight:600}
td.num,th.num{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}tr.final td{background:var(--final)}
.cite{display:block;font-size:12px;color:var(--muted)}td a{color:var(--accent);text-decoration:none}td a:hover{text-decoration:underline}
.tag{font-size:12px;border-radius:4px;padding:1px 6px;white-space:nowrap}.tag.pub{background:var(--pub);color:var(--pubink)}
.tag.t1{background:var(--t1);color:var(--t1ink)}.tag.ours{background:var(--ours);color:var(--oursink)}
.proto{font-size:13px;color:var(--muted);margin:0 0 6px}ol.fn{font-size:13px;color:var(--muted);padding-left:20px}
@media (max-width:560px){.bar{grid-template-columns:110px 1fr 48px}}
</style></head><body><main>
<h1>Emotion Classification with Qwen3.5-4B and LoRA</h1>
<p class="sub">Four datasets: SMP2020-EWECT general and COVID-19 Weibo (Chinese), GoEmotions (English), WRIME ver2 (Japanese).
One forward pass per text; the probabilities are a softmax over the answer word of each class.</p>
__BANNER__
<div class="card">
<div class="row"><div><label for="d">Dataset</label><select id="d"></select></div>
<div><label for="m">Model</label><select id="m"></select><div class="desc" id="md"></div></div></div>
<label for="t">Text</label><textarea id="t"></textarea>
<div class="ex" id="ex"></div>
<button id="go">Classify</button></div>
<div class="card result" id="out" hidden></div>
<p class="meta">Device: __DEVICE__. The first use of a model loads its weights, which takes a while.</p>
<h2>Test-set results and comparison with previous best published results</h2>
__TABLES__
<ol class="fn">__FOOTNOTES__</ol>
</main><script>
const DS=__DATA__,READY=__READY__;
const $=id=>document.getElementById(id),esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
DS.forEach(d=>{const o=document.createElement('option');o.value=d.id;o.textContent=d.title;$('d').appendChild(o)});
function cur(){return DS.find(d=>d.id===$('d').value)}
function setModel(){const d=cur(),m=d.models.find(x=>x.id===$('m').value);$('md').textContent=m?m.description+(m.available?'':' (weights not found)'):''}
function setDataset(){const d=cur();$('m').innerHTML='';d.models.forEach(m=>{const o=document.createElement('option');o.value=m.id;
o.textContent=m.name;o.disabled=!m.available;$('m').appendChild(o)});$('t').lang=d.lang;$('t').placeholder=d.placeholder;
$('ex').innerHTML='';d.examples.forEach(s=>{const e=document.createElement('span');e.lang=d.lang;e.textContent=s;e.onclick=()=>{$('t').value=s};$('ex').appendChild(e)});
$('out').hidden=true;setModel()}
$('d').onchange=setDataset;$('m').onchange=setModel;setDataset();
if(!READY){$('go').disabled=true;$('go').title='started with --no-models'}
$('go').onclick=async()=>{const b=$('go'),o=$('out'),text=$('t').value.trim(),d=cur();if(!text)return;
b.disabled=true;o.hidden=false;o.innerHTML='<p class="meta">Running…</p>';
try{const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({text,dataset:d.id,model:$('m').value})});const j=await r.json();if(!r.ok)throw new Error(j.error||r.status);
const p=j.probabilities,k=p.indexOf(Math.max(...p));
o.innerHTML='<h2>'+esc(d.labels[k])+'<small lang="zh">'+esc(d.words[k])+'</small></h2>'+p.map((v,i)=>'<div class="bar'+(i===k?' best':'')+
'"><span>'+esc(d.labels[i])+'<small lang="zh">'+esc(d.words[i])+'</small></span><div class="track"><div class="fill" style="width:'+
(100*v).toFixed(1)+'%"></div></div><span>'+(100*v).toFixed(1)+'%</span></div>').join('')+'<p class="meta">'+esc(j.model)+' · '+j.ms+' ms</p>'}
catch(e){o.innerHTML='<p class="err">Error: '+esc(e.message)+'</p>'}finally{b.disabled=!READY}};
</script></body></html>'''

PLACEHOLDER = {'usual': 'Type a Chinese Weibo post', 'virus': 'Type a Chinese Weibo post about the COVID-19 epidemic',
               'goemotions': 'Type an English comment', 'wrime': 'Type a Japanese social-media post'}


def build_page(args, t1, models):
    ref = {key: (read_json(ROOT / 'reference' / f) if (ROOT / 'reference' / f).is_file() else None)
           for key, f in (('results', 'results.json'), ('published', 'published_results.json'),
                          ('thesis1', 'thesis1_results.json'), ('edom18', 'wrime_edom18_protocol.json'))}
    data = [dict(id=ds, title=domain(ds).title, labels=domain(ds).labels, words=domain(ds).words, lang=LANG[ds],
                 placeholder=PLACEHOLDER[ds], examples=EXAMPLES[ds], models=models[ds]) for ds in DATASETS]
    banner = '' if not args.no_models else \
        '<p class="note">Started with --no-models: classification is off; the results tables below are complete.</p>'
    device = 'none (--no-models)' if args.no_models else \
        ('CUDA GPU if available, else Apple-silicon GPU (mps), else CPU' if args.device == 'auto' else args.device)
    page = (PAGE.replace('__DATA__', json.dumps(data, ensure_ascii=False).replace('</', '<\\/'))
            .replace('__READY__', 'false' if args.no_models else 'true')
            .replace('__BANNER__', banner).replace('__DEVICE__', esc(device))
            .replace('__FOOTNOTES__', ''.join(f'<li>{f}</li>' for f in FOOTNOTES))
            .replace('__TABLES__', results_section(ref, t1)))
    return page.encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description='Local web page for the models and their test results.')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--device', choices=DEVICE_CHOICES, default='auto',
                        help='auto = cuda, else Apple-silicon mps, else cpu')
    parser.add_argument('--threads', type=int, help='CPU threads for --device cpu')
    parser.add_argument('--model-dir', type=Path, help='Qwen3.5-4B folder (default: models/Qwen3.5-4B)')
    parser.add_argument('--no-models', action='store_true', help='serve only the page and the results tables')
    args = parser.parse_args()
    t1 = load_thesis1()
    models = demo_models(t1)
    allowed = {ds: {m['id'] for m in rows if m['available']} for ds, rows in models.items()}
    names = {ds: {m['id']: m['name'] for m in rows} for ds, rows in models.items()}
    page = build_page(args, t1, models)
    engine = None if args.no_models else Engine(args, t1)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, body, kind):
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def reply_json(self, code, obj):
            self.reply(code, json.dumps(obj, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8')

        def do_GET(self):
            if self.path in ('/', '/index.html'):
                self.reply(200, page, 'text/html; charset=utf-8')
            else:
                self.reply(404, b'not found', 'text/plain')

        def do_POST(self):
            if self.path != '/api/predict':
                return self.reply(404, b'not found', 'text/plain')
            if engine is None:
                return self.reply_json(503, dict(error='the server was started with --no-models'))
            try:
                req = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))).decode('utf-8'))
                text, ds, model = str(req.get('text', '')).strip(), req.get('dataset'), req.get('model')
                if not text or ds not in allowed or model not in allowed[ds]:
                    raise ValueError('need text, a dataset (' + ', '.join(allowed) + ') and one of its available models')
                start = time.time()
                p = engine.predict(ds, model, text)
                k = int(p.argmax())
                self.reply_json(200, dict(probabilities=[float(v) for v in p], label=domain(ds).labels[k],
                                          answer_word=domain(ds).words[k], model=names[ds][model],
                                          ms=int(1000 * (time.time() - start))))
            except (Exception, SystemExit) as exc:          # report to the page instead of dropping the connection
                self.reply_json(400, dict(error=str(exc)))

        def log_message(self, *a):
            pass

    print(f'open http://{args.host}:{args.port}  (Ctrl+C to stop)' + ('  [--no-models]' if args.no_models else ''), flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
