"""Rebuild LoRA soups from their members with numpy (no torch) and compare them with the shipped soups.

Exact uniform soup ('exact'): n rank-8 adapters -> one rank-8n adapter whose update equals the mean of the n updates.
  A_soup = concat(A_1..A_n) along the rank axis, B_soup = concat(B_1..B_n) / n, lora_alpha = 16n (alpha / r stays 2),
  so B_soup @ A_soup = mean_i(B_i @ A_i) exactly. This is the construction of every final model in the package.
Factor average ('naive', usual-domain ablation only): A and B are averaged separately (r = 8, alpha = 16); this is
  not the mean of the updates.

  python tools/build_soup.py --domain usual --out my_soup                       # the six SD members (final model)
  python tools/build_soup.py --domain goemotions --kind a --out my_a_soup       # the three A seeds
  python tools/build_soup.py --members DIR1 DIR2 ... --out my_soup              # any rank-8 adapters
  python tools/build_soup.py --domain wrime --out my_soup --compare weights/wrime/sd_uniform6_soup
  python tools/build_soup.py --check-all                                        # rebuild every shipped soup, bit for bit

A shipped soup is a folder under weights/ with a members.json file that names its members.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from common import BASELINES, DATASETS, DOMAINS, MEMBERS, MODEL_NAMES, ROOT, model_path

KINDS = {'sd': MEMBERS, 'a': BASELINES}


def load_adapter(path):
    from safetensors.numpy import load_file
    path = Path(path)
    conf = json.loads((path / 'adapter_config.json').read_text(encoding='utf-8'))
    return conf, load_file(str(path / 'adapter_model.safetensors'))


def build(member_dirs, out_dir, method='exact', divide=False):
    """divide=True scales B by division by n instead of multiplication by 1/n (the two differ only in float32
    rounding; one early soup, weights/usual/a_uniform3_soup, was written with the division)."""
    from safetensors.numpy import save_file
    member_dirs = [Path(d) for d in member_dirs]
    loaded = [load_adapter(d) for d in member_dirs]
    confs, states = [c for c, _ in loaded], [s for _, s in loaded]
    if any(c['r'] != 8 or c['lora_alpha'] != 16 or c.get('use_dora') or c.get('use_rslora') for c in confs):
        raise SystemExit('every member must be a plain rank-8 LoRA with lora_alpha 16')
    keys = sorted(k for k in states[0] if '.lora_A.' in k)
    if any(sorted(s) != sorted(states[0]) for s in states):
        raise SystemExit('members target different modules')
    if any(k for k in states[0] if '.lora_A.' not in k and '.lora_B.' not in k):
        raise SystemExit('unexpected tensors in the adapters')
    n = len(states)
    merged = {}
    if method == 'exact':
        for k in keys:
            kb = k.replace('.lora_A.', '.lora_B.')
            merged[k] = np.ascontiguousarray(np.concatenate([s[k].astype(np.float32) for s in states], 0))
            cat = np.concatenate([s[kb].astype(np.float32) for s in states], 1)
            merged[kb] = np.ascontiguousarray(cat / np.float32(n) if divide else cat * np.float32(1.0 / n))
        conf = dict(confs[0], r=8 * n, lora_alpha=16 * n)
    elif method == 'naive':
        merged = {k: np.mean([s[k].astype(np.float64) for s in states], 0).astype(np.float32) for k in sorted(states[0])}
        conf = dict(confs[0])
    else:
        raise SystemExit(f'unknown method {method!r}')
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    save_file(merged, str(out_dir / 'adapter_model.safetensors'))
    (out_dir / 'adapter_config.json').write_text(json.dumps(conf, indent=2), encoding='utf-8')
    (out_dir / 'members.json').write_text(json.dumps([d.name for d in member_dirs], indent=1), encoding='utf-8')
    return out_dir


def compare(a_dir, b_dir):
    """(bit-for-bit identical, largest absolute tensor difference, largest |dB@A| on 12 modules)."""
    _, a = load_adapter(a_dir)
    _, b = load_adapter(b_dir)
    if sorted(a) != sorted(b) or any(a[k].shape != b[k].shape or a[k].dtype != b[k].dtype for k in a):
        return False, float('inf'), float('inf')
    identical = all(np.array_equal(a[k], b[k]) for k in a)
    gap = max(float(np.abs(a[k].astype(np.float64) - b[k]).max()) for k in a)
    keys = sorted(k for k in a if '.lora_A.' in k)[:12]
    update = max(float(np.abs(a[k.replace('.lora_A.', '.lora_B.')].astype(np.float64) @ a[k]
                              - b[k.replace('.lora_A.', '.lora_B.')].astype(np.float64) @ b[k]).max()) for k in keys)
    return identical, gap, update


def shipped_soups():
    """Every folder under weights/ that holds adapter weights and a members.json."""
    return sorted(p.parent for p in (ROOT / 'weights').rglob('members.json') if (p.parent / 'adapter_model.safetensors').is_file())


def soup_recipe(soup_dir):
    """(member folders, method) of a shipped soup, from its members.json.

    members.json is a list of member names (or an object with 'members' and optionally 'method'). A name is a model
    name of the soup's domain (SD_s42, A_s43, Afold1, ...), a sibling folder, or a package-relative path. The method
    follows from the soup's rank: 8 x n for the exact soup, 8 for the factor average of n > 1 members."""
    soup_dir = Path(soup_dir)
    spec = json.loads((soup_dir / 'members.json').read_text(encoding='utf-8'))
    names = spec if isinstance(spec, list) else spec['members']
    rank = json.loads((soup_dir / 'adapter_config.json').read_text(encoding='utf-8'))['r']
    method = spec.get('method') if isinstance(spec, dict) and spec.get('method') else \
        ('exact' if rank == 8 * len(names) else 'naive' if rank == 8 else None)
    if method is None:
        raise SystemExit(f'{soup_dir.relative_to(ROOT).as_posix()}: rank {rank} does not fit {len(names)} members')
    rel = soup_dir.relative_to(ROOT / 'weights').parts
    dom = rel[1] if rel[0] == 'ablations' else rel[0]
    weights_of = {d.weights: key for key, d in DOMAINS.items() if key in DATASETS}
    out = []
    for name in names:
        if name in MODEL_NAMES and dom in weights_of:
            out.append(model_path(weights_of[dom], name))
        elif (soup_dir.parent / name / 'adapter_config.json').is_file():
            out.append(soup_dir.parent / name)
        elif (ROOT / name / 'adapter_config.json').is_file():
            out.append(ROOT / name)
        else:
            raise SystemExit(f'{soup_dir.relative_to(ROOT).as_posix()}/members.json: member {name!r} not found')
    return out, method


def check_all(verbose=True):
    """Rebuild every shipped soup in a temporary folder; returns a list of (soup, identical, gap)."""
    results = []
    tmp = Path(tempfile.mkdtemp(prefix='soup_check_'))
    try:
        for soup in shipped_soups():
            members, method = soup_recipe(soup)
            identical, gap, _ = compare(build(members, tmp / 'soup', method), soup)
            how = method
            if not identical and method == 'exact':
                identical, gap2, _ = compare(build(members, tmp / 'soup', method, divide=True), soup)
                how, gap = ('exact, B / n', gap2) if identical else (how, gap)
            results.append((soup.relative_to(ROOT).as_posix(), identical, gap))
            if verbose:
                print(f'  {"ok  " if identical else "FAIL"} {soup.relative_to(ROOT).as_posix()}: {len(members)} members, '
                      f'{how}, bit-for-bit identical = {identical}, max |difference| = {gap:.1e}', flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return results


def main():
    parser = argparse.ArgumentParser(description='Rebuild LoRA soups from their members.')
    parser.add_argument('--domain', choices=list(DATASETS), help='use the shipped members of this domain')
    parser.add_argument('--kind', choices=list(KINDS), default='sd', help='sd: the six SD models (final model); a: the three A seeds')
    parser.add_argument('--members', nargs='+', help='adapter folders')
    parser.add_argument('--method', choices=['exact', 'naive'], default='exact')
    parser.add_argument('--divide', action='store_true', help='exact soup with B divided by n (see build())')
    parser.add_argument('--out', help='output folder')
    parser.add_argument('--compare', help='a soup folder to compare the result with')
    parser.add_argument('--check-all', action='store_true', help='rebuild every shipped soup and compare bit for bit')
    args = parser.parse_args()
    if args.check_all:
        results = check_all()
        bad = [s for s, identical, _ in results if not identical]
        print(f'{len(results)} soups rebuilt, {len(bad)} differ' + (f': {bad}' if bad else ''))
        raise SystemExit(1 if bad or not results else 0)
    if bool(args.domain) == bool(args.members) or not args.out:
        parser.error('give --out and either --domain or --members')
    members = args.members or [model_path(args.domain, m) for m in KINDS[args.kind]]
    out = build(members, args.out, args.method, args.divide)
    print('soup written to', out, '| members:', ', '.join(Path(m).name for m in members))
    if args.compare:
        identical, gap, update = compare(out, args.compare)
        print(f'compared with {args.compare}: bit-for-bit identical = {identical}, max |difference| = {gap:.1e}, '
              f'max |dB@A| on 12 modules = {update:.1e}')


if __name__ == '__main__':
    main()
