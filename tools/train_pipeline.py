"""Full training recipe of the thesis for one domain; training needs an NVIDIA GPU with about 16 GB.

  python tools/train_pipeline.py --domain goemotions plan                 # list the steps, their state and time
  python tools/train_pipeline.py --domain goemotions run                  # run every pending step in order (resumable)
  python tools/train_pipeline.py --domain goemotions run --from SD_s42    # start at a step (reuses the shipped targets)
  python tools/train_pipeline.py --domain usual run --skip-baselines      # skip A_s43, A_s44 and the A soup
  python tools/train_pipeline.py --domain virus check-targets             # CPU: rebuild the shipped soft targets

Domains: usual, virus, goemotions, wrime.
Steps (run names are <domain>_<step>; everything is written to runs/, shipped files are never changed):
   prepare               wrime only: python tools/build_wrime.py (the other domains ship their data)
   A_s42, A_s43, A_s44   direct LoRA fine-tuning (A), three seeds
   Afold1, Afold2        the A recipe on one stratified half of the training rows each (folds/<domain>_fold<k>_spec.json)
   heldout:Afold1/2      each half model scores the half it did not see
   targets:sd            q = 0.7 x gold + 0.3 x out-of-fold probabilities; rows whose exact text is in both halves
                         keep the hard gold label -> runs/targets/<domain>_targets_sd.json
   SD_s42/43/44, SDfold1/2   the same recipe on the soft targets
   heldout:SDfold1/2, targets:sd_gen2, SDgen2   second generation: the SD half models are the teachers
   soup:sd_uniform6      exact uniform soup of the six SD models (final model), scored on dev and test
   soup:a_uniform3       exact uniform soup of the three A seeds, scored on dev and test
Training is reasoning_training.py with configs/<domain>_<step>.json. SD steps use the targets built by this pipeline
when they exist (runs/targets/) and the shipped thesis targets (targets/) otherwise; the config actually used is
written to runs/configs/. The test split is never used to build anything; it is only scored.
"""
import argparse
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import (BASELINES, MEMBERS, ROOT, domain, load_split, metrics, prediction_rows, print_metrics,
                    read_json, sha256, write_json)

PIPELINE_DOMAINS = ('usual', 'virus', 'goemotions', 'wrime')
TEACHERS = {'sd': ('Afold1', 'Afold2'), 'sd_gen2': ('SDfold1', 'SDfold2')}
TARGET_KINDS = {'sd': 'self-distillation 0.7 gold + 0.3 out-of-fold A (hard gold on cross-fold duplicates)',
                'sd_gen2': 'second-generation self-distillation: 0.7 gold + 0.3 out-of-fold SD half model '
                           '(hard gold on cross-fold duplicates)'}
SOUP_MEMBERS = {'sd_uniform6': MEMBERS, 'a_uniform3': BASELINES}
OPTIONAL = {'A_s43', 'A_s44', 'soup:a_uniform3'}          # not needed by the final model (--skip-baselines)


def steps(dom):
    out = [('prepare', dom)] if dom == 'wrime' else []
    out += [('train', f'A_s{s}') for s in (42, 43, 44)]
    out += [('train', 'Afold1'), ('train', 'Afold2'), ('heldout', 'Afold1'), ('heldout', 'Afold2'), ('targets', 'sd')]
    out += [('train', f'SD_s{s}') for s in (42, 43, 44)]
    out += [('train', 'SDfold1'), ('train', 'SDfold2'), ('heldout', 'SDfold1'), ('heldout', 'SDfold2'),
            ('targets', 'sd_gen2'), ('train', 'SDgen2'), ('soup', 'sd_uniform6'), ('soup', 'a_uniform3')]
    return out


def label(kind, name):
    return name if kind == 'train' else f'{kind}:{name}'


def run_dir(dom, name):
    return ROOT / 'runs' / f'{dom}_{name}'


def fresh_targets(dom, kind):
    return ROOT / 'runs' / 'targets' / f'{dom}_targets_{kind}.json'


def shipped_targets(dom, kind):
    return ROOT / 'targets' / f'{dom}_targets_{kind}.json'


def fold_spec(dom, k):
    return read_json(ROOT / 'folds' / f'{dom}_fold{k}_spec.json')


def fold_of(dom):
    return {int(k): v for k, v in read_json(ROOT / 'folds' / f'{dom}_fold_assignment.json').items()}


def data_ready(dom):
    train = domain(dom).folder('train') / 'train.tsv'
    return train.is_file() and sha256(train) == fold_spec(dom, 1)['source_train_sha256']


def done(dom, kind, name):
    if kind == 'prepare':
        return data_ready(dom)
    if kind == 'train':
        r = run_dir(dom, name) / 'results.json'
        return r.is_file() and read_json(r).get('status') == 'complete'
    if kind == 'heldout':
        return (run_dir(dom, name) / 'train_heldout_predictions.json').is_file()
    if kind == 'targets':
        path = fresh_targets(dom, name)
        return path.is_file() and read_json(path).get('built_by') == 'train_pipeline'
    return (run_dir(dom, f'{name}_soup') / 'results.json').is_file()


def thesis_hours(dom, name):
    """Training time of the thesis run (RTX 4080) from logs/<domain>/<name>/results.json, in hours."""
    path = ROOT / 'logs' / dom / name / 'results.json'
    if not path.is_file():
        return None
    r = read_json(path)
    seconds = r.get('train_seconds') or sum(h.get('train_seconds', 0) for h in r.get('history', []))
    return seconds / 3600 if seconds else None


# ------------------------------------------------------------------ steps
def prepare(dom):
    if dom == 'wrime':
        subprocess.run([sys.executable, str(ROOT / 'tools' / 'build_wrime.py')], cwd=ROOT, check=True)
    if not data_ready(dom):
        raise SystemExit(f'data for {dom} does not match the training data of the thesis (fold spec hash)')


def train_rows(dom):
    """(training rows, SHA-256 of the training file)."""
    return load_split(dom, 'train'), sha256(domain(dom).folder('train') / 'train.tsv')


def config_for(dom, name):
    """configs/<domain>_<name>.json, with the pipeline's own targets if they exist; saved to runs/configs/."""
    cfg = read_json(ROOT / 'configs' / f'{dom}_{name}.json')
    if 'label_targets_file' in cfg:
        kind = 'sd_gen2' if cfg['label_targets_file'].endswith('_sd_gen2.json') else 'sd'
        fresh = fresh_targets(dom, kind)
        cfg['label_targets_file'] = fresh.relative_to(ROOT).as_posix() if fresh.is_file() else cfg['label_targets_file']
        print(f'   soft targets: {cfg["label_targets_file"]}', flush=True)
    path = ROOT / 'runs' / 'configs' / f'{dom}_{name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf-8')
    return path


def train(dom, name):
    print(f'== train {dom}_{name}', flush=True)
    if not (ROOT / 'model_manifest.json').is_file():
        raise SystemExit('model_manifest.json (file hashes of the base model, read by reasoning_training.py) is missing '
                         'from the package root')
    cfg = config_for(dom, name)
    subprocess.run([sys.executable, str(ROOT / 'reasoning_training.py'), '--config', str(cfg)], cwd=ROOT, check=True)


def new_scorer():
    from scorer import Scorer
    return Scorer(device='auto')         # cuda, else Apple-silicon mps, else cpu (scoring only; training needs CUDA)


def free(scorer):
    import torch
    del scorer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_run_prompt(dom, name):
    d = domain(dom)
    results = read_json(run_dir(dom, name) / 'results.json')
    prov = results['provenance']
    prefix = prov['config'].get('text_prefix')
    prefix = '微博文本：\n' if prefix is None else prefix
    if prov['instruction'] != d.instruction or list(results['answer_texts']) != d.words or prefix != d.prefix:
        raise RuntimeError(f'{dom}_{name} was trained with a different prompt than tools/common.py states')


def reproduces(scorer, dom, name, n, floor):
    rows = load_split(dom, 'dev')[:n]
    ref = [p['predicted'] for p in read_json(run_dir(dom, name) / 'dev_predictions.json')[:n]]
    agree = float((scorer.score([r['text'] for r in rows], dom).argmax(1) == np.array(ref)).mean())
    if agree < floor:
        raise SystemExit(f'scoring does not reproduce {dom}_{name} dev predictions (agreement {agree:.4f})')
    return agree


def heldout(dom, name):
    """Score the training rows that the half model's fold spec dropped (the other half)."""
    print(f'== held-out scoring {dom}_{name}', flush=True)
    check_run_prompt(dom, name)
    spec = fold_spec(dom, 1 if name.endswith('1') else 2)
    if spec['source_train_sha256'] != sha256(domain(dom).folder('train') / 'train.tsv'):
        raise RuntimeError('fold spec was built from different training data')
    held = set(spec['drop'])
    rows = [r for r in load_split(dom, 'train') if r['row_index'] in held]
    scorer = new_scorer()
    scorer.use(run_dir(dom, name) / 'best_adapter')
    agree = reproduces(scorer, dom, name, 256, 0.99)
    P = scorer.score([r['text'] for r in rows], dom, progress=True)
    free(scorer)
    write_json(run_dir(dom, name) / 'train_heldout_predictions.json', prediction_rows(rows, P))
    y = np.array([r['label'] for r in rows])
    print(f'{dom}_{name}: held-out rows {len(rows)}, accuracy {float((P.argmax(1) == y).mean()):.4f}, dev reproduction {agree}')


def soft_targets(dom, kind, heldout_by_fold, rows=None, train_sha=None):
    """0.7 gold + 0.3 out-of-fold distribution; rows whose exact text sits in both folds keep hard gold.
    heldout_by_fold: {1: predictions of the fold-1 model on fold-2 rows, 2: ...} (lists of dicts).
    rows / train_sha: the training rows and the SHA-256 of their file (default: read from data/)."""
    d = domain(dom)
    if rows is None:
        rows = load_split(dom, 'train')
        train_sha = sha256(d.folder('train') / 'train.tsv')
    y = np.array([r['label'] for r in rows])
    folds = fold_of(dom)
    oof = np.full((len(y), d.k), np.nan)
    for k, preds in heldout_by_fold.items():
        for p in preds:
            i = p['row_index']
            if folds[i] == k:
                raise RuntimeError(f'fold-{k} model scored a row of its own training fold')
            if p['gold'] != y[i]:
                raise RuntimeError('held-out predictions are not aligned with the training labels')
            oof[i] = p['probabilities']
    if np.isnan(oof).any():
        raise RuntimeError('some training rows have no out-of-fold prediction')
    oof = np.clip(oof, 1e-6, None)
    oof /= oof.sum(1, keepdims=True)
    folds_by_text = {}
    for r in rows:
        folds_by_text.setdefault(r['text'], set()).add(folds[r['row_index']])
    S = np.array([len(folds_by_text[r['text']]) > 1 for r in rows])
    eye = np.eye(d.k)
    q = 0.7 * eye[y] + 0.3 * oof
    q[S] = eye[y[S]]
    return dict(source_train_sha256=train_sha, kind=TARGET_KINDS[kind],
                oof_accuracy=float((oof.argmax(1) == y)[~S].mean()), cross_fold_duplicates=int(S.sum()),
                rows={str(i): dict(q=[float(v) for v in q[i] / q[i].sum()]) for i in range(len(y))})


def compare_targets(a, b):
    """(identical, largest |q difference|) of two target payloads."""
    if a['source_train_sha256'] != b['source_train_sha256'] or a['rows'].keys() != b['rows'].keys():
        return False, float('inf')
    gap = max(abs(u - v) for i in a['rows'] for u, v in zip(a['rows'][i]['q'], b['rows'][i]['q']))
    same = (gap == 0 and a['cross_fold_duplicates'] == b['cross_fold_duplicates'] and a['oof_accuracy'] == b['oof_accuracy'])
    return same, gap


def targets(dom, kind):
    teachers = TEACHERS[kind]
    preds = {k: read_json(run_dir(dom, t) / 'train_heldout_predictions.json') for k, t in zip((1, 2), teachers)}
    payload = soft_targets(dom, kind, preds)
    payload.update(built_by='train_pipeline', teachers=[f'{dom}_{t}' for t in teachers])
    write_json(fresh_targets(dom, kind), payload)
    _, gap = compare_targets(payload, read_json(shipped_targets(dom, kind)))
    print(f'{dom} targets {kind}: teacher accuracy {payload["oof_accuracy"]:.4f}, cross-fold duplicates '
          f'{payload["cross_fold_duplicates"]}; largest difference from the shipped thesis targets {gap:.4f}')


def check_targets(dom, verbose=True):
    """CPU: rebuild the shipped soft targets from the shipped held-out predictions of the teachers
    (reference/predictions/<domain>/<teacher>_train_heldout.json); returns {kind: identical}."""
    out = {}
    rows, train_sha = train_rows(dom)
    for kind, teachers in TEACHERS.items():
        preds = {k: read_json(ROOT / 'reference' / 'predictions' / dom / f'{t}_train_heldout.json')
                 for k, t in zip((1, 2), teachers)}
        payload = soft_targets(dom, kind, preds, rows, train_sha)
        shipped = read_json(shipped_targets(dom, kind))
        out[kind], gap = compare_targets(payload, shipped)
        if verbose:
            print(f'  {"ok  " if out[kind] else "FAIL"} {dom} targets {kind}: {len(payload["rows"])} rows rebuilt from '
                  f'{" + ".join(teachers)}, identical = {out[kind]}, max |difference| = {gap:.1e}, teacher accuracy '
                  f'{payload["oof_accuracy"]:.4f}, {payload["cross_fold_duplicates"]} cross-fold duplicates kept hard', flush=True)
    return out


def soup(dom, name):
    from build_soup import build
    members = SOUP_MEMBERS[name]
    for m in members:
        check_run_prompt(dom, m)
    out = run_dir(dom, f'{name}_soup')
    scorer = new_scorer()
    scorer.use(build([run_dir(dom, members[0]) / 'best_adapter'], out / 'probe'), name='probe')
    agree = reproduces(scorer, dom, members[0], 10 ** 9, 0.999)       # a one-member soup must reproduce its member
    scorer.use(build([run_dir(dom, m) / 'best_adapter' for m in members], out / 'adapter'), name='soup')
    scorer.drop('probe')
    shutil.rmtree(out / 'probe', ignore_errors=True)
    d = domain(dom)
    R = dict(domain=dom, members=[f'{dom}_{m}' for m in members], reproduction_agreement=agree)
    for part in ('dev', 'test'):
        rows = load_split(dom, part)
        P = scorer.score([r['text'] for r in rows], dom, progress=True)
        write_json(out / f'{part}_predictions.json', prediction_rows(rows, P))
        R[part] = metrics([r['label'] for r in rows], P.argmax(1), d.k)
        print_metrics(R[part], f'[{dom} {name} soup / {part}]', d)
    free(scorer)
    write_json(out / 'results.json', R)


def main():
    parser = argparse.ArgumentParser(description='Training recipe of the thesis for one domain.')
    parser.add_argument('--domain', choices=PIPELINE_DOMAINS, required=True)
    parser.add_argument('action', choices=['plan', 'run', 'check-targets'])
    parser.add_argument('--from', dest='start', help='first step to run (a step name from plan)')
    parser.add_argument('--skip-baselines', action='store_true', help='skip A_s43, A_s44 and the A soup')
    args = parser.parse_args()
    dom = args.domain
    if args.action == 'check-targets':
        ok = check_targets(dom)
        raise SystemExit(0 if all(ok.values()) else 1)
    plan = steps(dom)
    names = [label(kind, name) for kind, name in plan]
    if args.action == 'plan':
        total = 0.0
        for i, ((kind, name), lab) in enumerate(zip(plan, names), 1):
            hours = thesis_hours(dom, name) if kind == 'train' else None
            total += hours or 0
            state = 'done' if done(dom, kind, name) else 'pending'
            print(f'{i:2d}. {lab:22s} {state:8s}' + (f' thesis run: {hours:5.2f} h' if hours else ''))
        print(f'training time of the thesis runs (one RTX 4080): {total:.1f} h in total (single runs varied by up to 2x '
              f'with machine load); held-out scoring and the two soups add a few GPU minutes per 1,000 rows')
        return
    first = 0
    if args.start:
        if args.start not in names:
            raise SystemExit(f'--from must be one of: {", ".join(names)}')
        first = names.index(args.start)
    for (kind, name), lab in list(zip(plan, names))[first:]:
        if args.skip_baselines and lab in OPTIONAL:
            continue
        if done(dom, kind, name):
            print(f'-- {lab}: done', flush=True)
            continue
        {'prepare': lambda: prepare(dom), 'train': lambda: train(dom, name), 'heldout': lambda: heldout(dom, name),
         'targets': lambda: targets(dom, name), 'soup': lambda: soup(dom, name)}[kind]()
    print('all steps finished')


if __name__ == '__main__':
    main()
