"""Predict the emotion of one or more texts from the command line.

  python tools/predict.py "排了三个小时的队，结果系统坏了，气死我了。"
  python tools/predict.py --domain virus --model A_s42 "小区又封了，什么时候才是个头啊。"
  python tools/predict.py --domain goemotions "I can't believe they cancelled the show again."
  python tools/predict.py --domain wrime "明日から旅行だ、楽しみすぎて眠れない！"
  python tools/predict.py --domain usual --file texts.txt        # one text per line

Use the weights of the dataset the text comes from: the class sets differ (6, 7 or 9 classes).
"""
import argparse
import sys
from pathlib import Path

from common import DATASETS, DEVICE_CHOICES, MODEL_HELP, domain, model_path


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description='Predict the emotion of texts.')
    parser.add_argument('texts', nargs='*')
    parser.add_argument('--file', type=Path, help='UTF-8 file with one text per line')
    parser.add_argument('--domain', choices=list(DATASETS), default='usual',
                        help='usual (general Weibo), virus (COVID-19 Weibo), goemotions (English Reddit) or '
                             'wrime (Japanese SNS)')
    parser.add_argument('--model', default='soup', help=MODEL_HELP)
    parser.add_argument('--device', choices=DEVICE_CHOICES, default='auto',
                        help='auto = cuda, else Apple-silicon mps, else cpu')
    parser.add_argument('--threads', type=int)
    parser.add_argument('--model-dir', type=Path)
    args = parser.parse_args()
    texts = list(args.texts)
    if args.file:
        texts += [line.strip() for line in args.file.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not texts:
        parser.error('give a text or --file')
    d = domain(args.domain)
    adapter = model_path(d, args.model)
    from scorer import Scorer
    scorer = Scorer(args.model_dir, device=args.device, threads=args.threads)
    scorer.use(adapter)
    for text, p in zip(texts, scorer.score(texts, d)):
        k = int(p.argmax())
        print(f'{d.labels[k]}  ' + '  '.join(f'{n} {v:.3f}' for n, v in zip(d.labels, p)) + f'  | {text}')


if __name__ == '__main__':
    main()
