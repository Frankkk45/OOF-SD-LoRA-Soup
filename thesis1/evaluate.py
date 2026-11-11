"""Evaluate the Thesis 1 models on data/usual dev and test: accuracy, Macro-F1, per-class scores, confusion matrix.

  python thesis1/evaluate.py                                  # all five models, dev and test
  python thesis1/evaluate.py --model roberta_bigru --split test
  python thesis1/evaluate.py --model all --split test --limit 200 --threads 8

If reference/predictions/usual/thesis1_<model>_<split>.json exists, each row is compared with it (label agreement
and largest probability difference). Those reference files were computed on CPU in float32 with this script. The
Thesis 1 notebooks evaluated on GPU under float16 autocast; their test scores are listed as expected_test in
reference/thesis1_results.json and can differ from a float32 run by a few near-tied rows.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import LABELS, ROOT, THESIS1_MODELS, load_model, metrics, pick_device, read_tsv  # noqa: E402

KEYS = [m['key'] for m in THESIS1_MODELS]


def print_metrics(m, title):
    print(f'\n{title}')
    print(f'  Accuracy {100 * m["accuracy"]:.2f}   Macro-F1 {100 * m["macro_f1"]:.2f}')
    print('  class      precision  recall     F1   support')
    for i, name in enumerate(LABELS):
        print(f'  {name:9s} {100 * m["precision"][i]:9.2f} {100 * m["recall"][i]:7.2f} {100 * m["f1"][i]:7.2f} '
              f'{m["support"][i]:7d}')
    print('  confusion matrix (rows = gold, columns = predicted, order ' + ' '.join(LABELS) + ')')
    for i, row in enumerate(m['confusion']):
        print(f'  {LABELS[i]:9s}' + ''.join(f'{v:6d}' for v in row))


def prediction_rows(labels, P):
    return [dict(row_index=i, gold=int(g), predicted=int(p.argmax()), probabilities=p.tolist())
            for i, (g, p) in enumerate(zip(labels, P))]


def compare(ref_path, P):
    """Label agreement and largest probability difference against a saved prediction file."""
    R = json.loads(Path(ref_path).read_text(encoding='utf-8'))[:len(P)]
    agree = int(sum(r['predicted'] == int(p.argmax()) for r, p in zip(R, P)))
    gap = float(np.max(np.abs(np.array([r['probabilities'] for r in R]) - P))) if R else 0.0
    return agree, len(R), gap


def evaluate(key, splits, device='cpu', batch_size=32, limit=None, root=None, data_dir=None, encoder_dir=None,
             weights_dir=None, autocast=False):
    """{split: (labels, probabilities, metrics)} for one model."""
    import torch
    root = Path(root) if root else ROOT
    data_dir = Path(data_dir) if data_dir else root / 'data' / 'usual'
    model = load_model(key, device=device, root=root, encoder_dir=encoder_dir, weights_dir=weights_dir)
    out = {}
    for split in splits:
        labels, texts = read_tsv(data_dir / f'{split}.tsv')
        labels, texts = labels[:limit], texts[:limit]
        if autocast and device.startswith('cuda'):
            with torch.autocast('cuda', dtype=torch.float16):
                P = model.predict_proba(texts, batch_size=batch_size)
        else:
            P = model.predict_proba(texts, batch_size=batch_size)
        out[split] = (labels, P, metrics(labels, P.argmax(1)))
    del model
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model', default='all', choices=['all'] + KEYS)
    parser.add_argument('--split', default='both', choices=['dev', 'test', 'both'])
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='auto = cuda, else Apple-silicon mps, else cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, help='CPU threads')
    parser.add_argument('--limit', type=int, help='score only the first N rows of each split')
    parser.add_argument('--autocast', action='store_true', help='float16 autocast on CUDA, as in the notebooks')
    parser.add_argument('--root', type=Path, help='package root (default: the folder above thesis1/)')
    parser.add_argument('--data-dir', type=Path, help='folder with dev.tsv and test.tsv (default: <root>/data/usual)')
    parser.add_argument('--encoder-dir', type=Path, help='default: <root>/models/chinese-roberta-wwm-ext')
    parser.add_argument('--weights-dir', type=Path, help='default: <root>/weights/thesis1')
    parser.add_argument('--save', type=Path, help='folder for thesis1_<model>_<split>.json prediction files')
    args = parser.parse_args()

    device = pick_device(args.device)
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    root = args.root or ROOT
    keys = KEYS if args.model == 'all' else [args.model]
    splits = ['dev', 'test'] if args.split == 'both' else [args.split]
    results_path = root / 'reference' / 'thesis1_results.json'
    expected = json.loads(results_path.read_text(encoding='utf-8')) if results_path.is_file() else {}
    print(f'device: {device}')
    summary = []
    for key in keys:
        start = time.time()
        res = evaluate(key, splits, device=device, batch_size=args.batch_size, limit=args.limit, root=root,
                       data_dir=args.data_dir, encoder_dir=args.encoder_dir, weights_dir=args.weights_dir,
                       autocast=args.autocast)
        name = next(m['display_name'] for m in THESIS1_MODELS if m['key'] == key)
        for split, (labels, P, m) in res.items():
            print_metrics(m, f'[{key} / {split}] {name}: {len(labels)} rows')
            ref = root / 'reference' / 'predictions' / 'usual' / f'thesis1_{key}_{split}.json'
            if ref.is_file():
                agree, n, gap = compare(ref, P)
                print(f'  vs {ref.relative_to(root).as_posix()}: same label on {agree}/{n} rows, '
                      f'max probability difference {gap:.2e}')
            exp = expected.get(key, {}).get('expected_test')
            if split == 'test' and exp and args.limit is None:
                print(f'  Thesis 1 notebook (GPU, float16 autocast): Accuracy {100 * exp["accuracy"]:.2f}   '
                      f'Macro-F1 {100 * exp["macro_f1"]:.2f}')
            if args.save:
                args.save.mkdir(parents=True, exist_ok=True)
                (args.save / f'thesis1_{key}_{split}.json').write_text(
                    json.dumps(prediction_rows(labels, P), ensure_ascii=False), encoding='utf-8')
            summary.append((key, split, m['accuracy'], m['macro_f1']))
        print(f'  ({time.time() - start:.0f} s)')
    if len(summary) > 1:
        print('\nmodel                          split   Accuracy  Macro-F1')
        for key, split, acc, f1 in summary:
            print(f'{key:30s} {split:5s} {100 * acc:9.2f} {100 * f1:9.2f}')


if __name__ == '__main__':
    main()
