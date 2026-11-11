"""Evaluate a model on a dev or test split: Accuracy, Macro-F1, Weighted-F1, per-class scores, confusion matrix.

  python tools/evaluate.py --domain usual --split test --model soup
  python tools/evaluate.py --domain both --split test --model soup            # usual + virus and the official score
  python tools/evaluate.py --domain goemotions --split dev --model A_s42 --limit 64 --device cpu
  python tools/evaluate.py --domain wrime --split test --model soup           # after: python tools/build_wrime.py
  python tools/evaluate.py --domain wrime_edom18 --model soup                 # after: python tools/build_wrime.py --protocol edom18

If reference/predictions/<domain>/<model>_<split>.json exists (the thesis run's own predictions), the script also
reports how many predicted labels agree with it and the largest probability difference.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import DEVICE_CHOICES, DOMAINS, MODEL_HELP, MODEL_NAMES, ROOT, domain, load_split, metrics, model_path, prediction_rows, print_metrics, write_json


def main():
    parser = argparse.ArgumentParser(description='Evaluate one model on one split.')
    parser.add_argument('--domain', choices=list(DOMAINS) + ['both'], required=True,
                        help='both = usual and virus, followed by the official SMP2020-EWECT score')
    parser.add_argument('--split', choices=['dev', 'test'], default='test')
    parser.add_argument('--model', default='soup', help=MODEL_HELP)
    parser.add_argument('--limit', type=int, help='score only the first N rows (quick check)')
    parser.add_argument('--device', choices=DEVICE_CHOICES, default='auto',
                        help='auto = cuda, else Apple-silicon mps, else cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, help='CPU threads (cpu device only)')
    parser.add_argument('--model-dir', type=Path, help='Qwen3.5-4B folder (default: models/Qwen3.5-4B)')
    parser.add_argument('--save', type=Path, help='folder for <domain>_<model>_<split>_predictions.json')
    args = parser.parse_args()
    keys = ['usual', 'virus'] if args.domain == 'both' else [args.domain]
    for key in keys:                                  # fail early, before the base model is loaded
        load_split(key, args.split)
        model_path(key, args.model)
    from scorer import Scorer
    scorer = Scorer(args.model_dir, device=args.device, threads=args.threads)
    print(f'device: {scorer.device}')
    macro = {}
    for key in keys:
        d = domain(key)
        rows = load_split(d, args.split)[:args.limit]
        scorer.use(model_path(d, args.model), name=f'{key}:{args.model}')
        start = time.time()
        P = scorer.score([r['text'] for r in rows], d, batch_size=args.batch_size, progress=len(rows) > 400)
        m = metrics([r['label'] for r in rows], P.argmax(1), d.k)
        macro[key] = m['macro_f1']
        print_metrics(m, f'[{key} / {args.split} / {args.model}] {len(rows)} rows, {time.time() - start:.0f} s', d)
        ref = ROOT / 'reference' / 'predictions' / key / f'{args.model}_{args.split}.json'
        if ref.is_file():
            R = json.loads(ref.read_text(encoding='utf-8'))[:len(rows)]
            if [r['gold'] for r in R] != [r['label'] for r in rows]:
                raise RuntimeError(f'{ref.relative_to(ROOT).as_posix()} is not aligned with the split')
            agree = float(np.mean([r['predicted'] == int(p.argmax()) for r, p in zip(R, P)]))
            gap = float(np.max(np.abs(np.array([r['probabilities'] for r in R]) - P)))
            print(f'  vs thesis predictions ({ref.relative_to(ROOT).as_posix()}): label agreement {100 * agree:.2f}%, '
                  f'max probability difference {gap:.4f}')
        if args.save:
            tag = args.model if args.model in MODEL_NAMES else Path(args.model).name
            write_json(args.save / f'{key}_{tag}_{args.split}_predictions.json', prediction_rows(rows, P))
        scorer.drop(f'{key}:{args.model}')
    if len(keys) == 2:
        print(f'\nOfficial SMP2020-EWECT score = (usual Macro-F1 + virus Macro-F1) / 2 = '
              f'{100 * (macro["usual"] + macro["virus"]) / 2:.2f}' + ('   (partial rows: --limit given)' if args.limit else ''))


if __name__ == '__main__':
    main()
