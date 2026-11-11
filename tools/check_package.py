"""CPU-only self-check of the package (no GPU, no torch; the base model is only hashed, never loaded).

  python tools/check_package.py                    # every check below
  python tools/check_package.py --skip-hashes      # everything except the SHA-256 of each file (faster)
  python tools/check_package.py --write-manifest   # rewrite MANIFEST.sha256 (after changing files on purpose)

1. files: every file matches MANIFEST.sha256; no processed WRIME split is listed (licence);
2. data: row counts and labels of every split (WRIME is rebuilt in a temporary folder if data/wrime has no splits);
3. folds: re-derived from the training labels (stratified, seed 20261017) and equal to the shipped fold files;
4. soft targets: rebuilt from the shipped held-out predictions of the teachers and identical to targets/;
5. configs: only keys the training script knows, every referenced file exists;
6. soups: every shipped soup rebuilt from its members (tools/build_soup.py) and bit-for-bit identical;
7. reference predictions: Accuracy / Macro-F1 recomputed and equal to reference/results.json;
8. Thesis 1: weight hashes and recomputed test scores.
"""
import argparse
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from common import DATA_OVERRIDE, DOMAINS, ROOT, domain, load_split, metrics, read_json, sha256

FOLD_SEED = 20261017
MANIFEST = ROOT / 'MANIFEST.sha256'
# Files written by the tools themselves; they are not part of the shipped package.
GENERATED = ('runs/', 'data/wrime_edom18/', 'data/wrime/train.tsv', 'data/wrime/dev.tsv', 'data/wrime/test.tsv')
PIPELINE_DOMAINS = ('usual', 'virus', 'goemotions', 'wrime')


class Report:
    def __init__(self):
        self.failed, self.skipped, self.passed = [], [], 0

    def ok(self, msg):
        self.passed += 1
        print('  ok    ' + msg, flush=True)

    def fail(self, msg):
        self.failed.append(msg)
        print('  FAIL  ' + msg, flush=True)

    def skip(self, msg):
        self.skipped.append(msg)
        print('  skip  ' + msg, flush=True)

    def check(self, cond, msg, detail=''):
        (self.ok if cond else self.fail)(msg + ('' if cond or not detail else f' ({detail})'))
        return cond


def package_files():
    """Every shipped file (package-relative POSIX path), without caches, generated files and the manifest."""
    out = []
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix in ('.pyc', '.tmp') or p == MANIFEST:
            continue
        rel = p.relative_to(ROOT).as_posix()
        if not rel.startswith(GENERATED):
            out.append(rel)
    return out


def write_manifest():
    files = package_files()
    MANIFEST.write_text(''.join(f'{sha256(ROOT / rel)}  {rel}\n' for rel in files), encoding='utf-8')
    print(f'MANIFEST.sha256 written: {len(files)} files')


def check_files(rep, skip_hashes):
    print('\n1. files')
    built = sorted(p.relative_to(ROOT).as_posix() for p in list((ROOT / 'data' / 'wrime').glob('*.tsv'))
                   + list((ROOT / 'data' / 'wrime_edom18').glob('*.tsv')))
    if built:
        print(f'  note  processed WRIME splits were built locally ({", ".join(built)}); the licence does not allow '
              'sharing them: leave them out when passing the package on')
    if not MANIFEST.is_file():
        rep.skip('MANIFEST.sha256 not found (write it with --write-manifest)')
        return
    entries = [line.split('  ', 1) for line in MANIFEST.read_text(encoding='utf-8').splitlines() if line.strip()]
    listed = {rel for _, rel in entries}
    wrime_text = [rel for rel in listed if rel.startswith(GENERATED[1:])]
    rep.check(not wrime_text, 'no processed WRIME split is listed in the manifest', f'{wrime_text[:3]}')
    missing = [rel for rel in listed if not (ROOT / rel).is_file()]
    extra = [rel for rel in package_files() if rel not in listed]
    rep.check(not missing, f'all {len(listed)} listed files are present', f'missing: {missing[:5]}')
    if extra:
        print(f'  note  {len(extra)} files are not listed in the manifest, e.g. {extra[:3]}')
    if skip_hashes:
        rep.skip('file hashes (--skip-hashes)')
        return
    bad = [rel for digest, rel in entries if (ROOT / rel).is_file() and sha256(ROOT / rel) != digest]
    rep.check(not bad, f'{len(entries) - len(missing)} files match MANIFEST.sha256', f'differ: {bad[:5]}')


def derive_folds(rows, k):
    rng = random.Random(FOLD_SEED)
    fold_of = {}
    for lab in range(k):
        idx = [r['row_index'] for r in rows if r['label'] == lab]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            fold_of[i] = 1 + j % 2
    return fold_of


def check_domain_inputs(rep, dom, rt, tp):
    """Data, folds, soft targets and configs of one training domain."""
    d = domain(dom)
    try:
        sizes = {part: len(load_split(dom, part)) for part in ('train', 'dev', 'test')}
        rep.ok('data rows ' + ' / '.join(f'{p} {n}' for p, n in sizes.items()))
        rows, train_sha = tp.train_rows(dom)
    except SystemExit as exc:
        rep.skip(f'{dom} data: {exc}')
        rows = None
    if rows is not None:
        fold_of = derive_folds(rows, d.k)
        shipped = {int(k): v for k, v in read_json(ROOT / 'folds' / f'{dom}_fold_assignment.json').items()}
        good = shipped == fold_of
        for k in (1, 2):
            spec = read_json(ROOT / 'folds' / f'{dom}_fold{k}_spec.json')
            good &= spec['source_train_sha256'] == train_sha and spec['drop'] == sorted(i for i, f in fold_of.items() if f != k)
            good &= spec['result_rows'] == sum(1 for f in fold_of.values() if f == k)
        rep.check(good, f'folds re-derived: {sum(v == 1 for v in fold_of.values())} / {sum(v == 2 for v in fold_of.values())} '
                        'rows, equal to folds/ (assignment, specs, training-file hash)')
        for kind, same in tp.check_targets(dom, verbose=False).items():
            rep.check(same, f'targets/{dom}_targets_{kind}.json rebuilt from the shipped held-out predictions of '
                            f'{" + ".join(tp.TEACHERS[kind])}: identical')
    cfgs = sorted((ROOT / 'configs').glob(f'{dom}_*.json'))
    bad = []
    for path in cfgs:
        cfg = read_json(path)
        if not set(cfg) <= set(rt.DEFAULTS) or cfg['run_name'] != path.stem or cfg['data_dir'] != f'data/{dom}':
            bad.append(path.name)
        for key in ('train_cleaning', 'label_targets_file'):
            if key in cfg and not (ROOT / cfg[key]).is_file():
                bad.append(f'{path.name}:{key}')
        if cfg.get('instruction_override', d.instruction) != d.instruction or cfg.get('text_prefix') not in (None, d.prefix) \
                or cfg.get('answer_words', d.words) != d.words or cfg.get('label_names', d.labels) != d.labels:
            bad.append(f'{path.name}:prompt')
    rep.check(len(cfgs) >= 11 and not bad, f'{len(cfgs)} configs valid (keys, paths, prompt equal to tools/common.py)', f'{bad[:5]}')


def check_reference(rep):
    print('\n7. reference predictions')
    results = read_json(ROOT / 'reference' / 'results.json')
    for dom in DOMAINS:
        folder = ROOT / 'reference' / 'predictions' / dom
        if dom not in results:
            continue
        d = domain(dom)
        gold, parts = {}, [p for p in ('dev', 'test') if p in d.splits]
        for part in parts:
            try:
                gold[part] = [r['label'] for r in load_split(dom, part)]
            except SystemExit:
                pass
        n, bad, unaligned = 0, [], []
        for model, by_split in results[dom].items():
            # '<model>_on_<other domain>': scored on the splits of another domain (cross-domain rows)
            other = model.rsplit('_on_', 1)[1] if '_on_' in model else None
            for part, want in by_split.items():
                path = folder / f'{model}_{part}.json'
                if part not in ('dev', 'test') or not isinstance(want, dict) or 'macro_f1' not in want:
                    continue
                if not path.is_file():
                    bad.append(f'{path.name} missing')
                    continue
                P = read_json(path)
                m = metrics([p['gold'] for p in P], [p['predicted'] for p in P], d.k)
                n += 1
                if abs(m['accuracy'] - want['accuracy']) > 1e-12 or abs(m['macro_f1'] - want['macro_f1']) > 1e-12:
                    bad.append(path.name)
                if other in DOMAINS:
                    if [p['gold'] for p in P] != [r['label'] for r in load_split(other, part)]:
                        unaligned.append(path.name)
                elif part in gold and [p['gold'] for p in P] != gold[part]:
                    unaligned.append(path.name)
        note = ', gold labels equal to the splits' if set(gold) == set(parts) else \
            ' (gold labels of the split not checked: build the data first)'
        rep.check(n and not bad and not unaligned, f'{dom}: {n} prediction files reproduce reference/results.json{note}',
                  f'differ: {bad[:4]}, not aligned: {unaligned[:4]}')
    path = ROOT / 'reference' / 'ablations.json'
    if path.is_file():
        abl = read_json(path)
        gold = {part: [r['label'] for r in load_split('usual', part)] for part in ('dev', 'test')}
        n, bad = 0, []
        for group in ('trained', 'soups'):
            for name, entry in (abl.get(group) or {}).items():
                for part in ('dev', 'test'):
                    want = entry.get(part)
                    if not isinstance(want, dict):
                        continue
                    pred = ROOT / 'reference' / 'predictions' / 'ablations' / 'usual' / f'{name}_{part}.json'
                    P = read_json(pred) if pred.is_file() else []
                    m = metrics([p['gold'] for p in P], [p['predicted'] for p in P], 6) if P else None
                    n += 1
                    if not m or abs(m['accuracy'] - want['accuracy']) > 1e-12 or abs(m['macro_f1'] - want['macro_f1']) > 1e-12 \
                            or [p['gold'] for p in P] != gold[part]:
                        bad.append(pred.name)
        rep.check(n and not bad, f'usual ablations: {n} prediction files reproduce reference/ablations.json', f'{bad[:4]}')
    path = ROOT / 'reference' / 'wrime_edom18_protocol.json'
    want = (read_json(path).get('manifest') or {}).get('test_tsv_sha256') if path.is_file() else None
    test = domain('wrime_edom18').folder('test') / 'test.tsv'
    if want and test.is_file():
        rep.check(sha256(test) == want, 'wrime_edom18 test split equals the one scored in reference/wrime_edom18_protocol.json')


def check_thesis1(rep):
    print('\n8. Thesis 1')
    weights = sorted((ROOT / 'weights' / 'thesis1').glob('*.pth'))
    if not weights:
        rep.skip('weights/thesis1 not found')
        return
    sys.path.insert(0, str(ROOT / 'thesis1'))
    try:
        from models import THESIS1_MODELS
    except ImportError as exc:
        rep.fail(f'thesis1/models.py: {exc}')
        return
    path = ROOT / 'reference' / 'thesis1_results.json'
    info = read_json(path) if path.is_file() else {}
    models = info.get('models', info) if isinstance(info, dict) else {}
    rep.check(sorted(m['file'] for m in THESIS1_MODELS) == sorted(p.name for p in weights),
              f'{len(weights)} Thesis 1 weight files, one per model in thesis1/models.py')
    for m in THESIS1_MODELS:
        entry = models.get(m['key'], {}) if isinstance(models, dict) else {}
        want = entry.get('sha256') or entry.get('weights_sha256')
        file = ROOT / 'weights' / 'thesis1' / m['file']
        if want:
            rep.check(file.is_file() and sha256(file) == want, f'{m["file"]}: SHA-256 equals reference/thesis1_results.json')
        else:
            rep.skip(f'{m["file"]}: no hash in reference/thesis1_results.json')
        pred = ROOT / 'reference' / 'predictions' / 'usual' / f'thesis1_{m["key"]}_test.json'
        test = entry.get('test') if isinstance(entry, dict) else None
        if pred.is_file() and isinstance(test, dict) and 'macro_f1' in test:
            P = read_json(pred)
            s = metrics([p['gold'] for p in P], [p['predicted'] for p in P], 6)
            rep.check(abs(s['accuracy'] - test['accuracy']) < 1e-12 and abs(s['macro_f1'] - test['macro_f1']) < 1e-12,
                      f'{m["key"]}: test predictions reproduce accuracy {100 * s["accuracy"]:.2f} / Macro-F1 {100 * s["macro_f1"]:.2f}')


def main():
    parser = argparse.ArgumentParser(description='CPU-only self-check of the package.')
    parser.add_argument('--skip-hashes', action='store_true', help='do not hash every file')
    parser.add_argument('--write-manifest', action='store_true', help='write MANIFEST.sha256 and exit')
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest()
        return
    rep = Report()
    check_files(rep, args.skip_hashes)
    tmp = Path(tempfile.mkdtemp(prefix='package_check_'))
    try:
        wrime_splits(rep, tmp)
        run_checks(rep)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f'\n{rep.passed} checks passed, {len(rep.failed)} failed, {len(rep.skipped)} skipped')
    for msg in rep.skipped:
        print('  skipped: ' + msg)
    if rep.failed:
        raise SystemExit('FAILED:\n  ' + '\n  '.join(rep.failed))
    print('all checks passed')


def wrime_splits(rep, tmp):
    """WRIME splits are not shipped (licence): when data/wrime or data/wrime_edom18 has not been built, rebuild them
    with tools/build_wrime.py into a temporary folder that is removed after the check."""
    for name, protocol in (('wrime', 'ours'), ('wrime_edom18', 'edom18')):
        if (ROOT / 'data' / name / 'test.tsv').is_file():
            continue
        out = tmp / name
        done = subprocess.run([sys.executable, str(ROOT / 'tools' / 'build_wrime.py'), '--protocol', protocol, '--out', str(out)],
                              cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        if done.returncode == 0:
            DATA_OVERRIDE[name] = out
            print(f'  note  {name} splits rebuilt from data/wrime/source in a temporary folder for this check')
        else:
            rep.fail(f'tools/build_wrime.py --protocol {protocol}: {(done.stderr or done.stdout).strip()[-300:]}')


def run_checks(rep):
    sys.path.insert(0, str(ROOT))
    import reasoning_training as rt
    import train_pipeline as tp
    from build_soup import check_all
    print('\n2-5. data, folds, soft targets, configs')
    for dom in PIPELINE_DOMAINS:
        print(f' == {dom}')
        try:
            check_domain_inputs(rep, dom, rt, tp)
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            rep.fail(f'{dom}: {type(exc).__name__}: {exc}')
    print('\n6. soups')
    soups = check_all(verbose=False)
    for soup, identical, gap in soups:
        rep.check(identical, f'{soup} rebuilt from its members: bit-for-bit identical', f'max |difference| {gap:.1e}')
    if not soups:
        rep.fail('no shipped soup found under weights/')
    check_reference(rep)
    check_thesis1(rep)
    results = read_json(ROOT / 'reference' / 'results.json')
    try:
        u, v = results['usual']['soup']['test']['macro_f1'], results['virus']['soup']['test']['macro_f1']
        print(f'\nofficial SMP2020-EWECT score of the final models: ({100 * u:.2f} + {100 * v:.2f}) / 2 = {100 * (u + v) / 2:.2f}')
    except KeyError:
        pass


if __name__ == '__main__':
    main()
