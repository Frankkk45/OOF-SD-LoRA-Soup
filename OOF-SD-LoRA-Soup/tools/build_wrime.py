"""Recreate the WRIME ver2 splits used in this package from the release file.

WRIME (Kajiwara et al., NAACL 2021; ver2: Suzuki et al., LREC 2022) is distributed under CC BY-NC-ND 4.0,
so the processed splits are not shipped. This script rebuilds them from the unchanged release file
data/wrime/source/wrime-ver2.tsv and checks every output against the SHA-256 of the files used in training.

Protocol 'ours' (default; used for training and for every WRIME result in the package)
  Labels: 0 neutral, 1 angry, 2 happy, 3 sad, 4 fear, 5 surprise, 6 disgust, 7 anticipation, 8 trust.
  For each of the eight Plutchik emotions take the exact mean of Reader1..3 (intensity 0-3). The released
  'Avg. Readers_<Emotion>' columns are rounded to integers and are not used by this rule.
    all eight means < 1.0                -> 0 neutral
    the strongest emotion is unique      -> that emotion
    otherwise                            -> row dropped ('tie')
  Text: every TAB, CR and LF becomes one space, then the text is stripped; empty text -> row dropped
  (checked before the label rule). The official Train/Dev/Test column is used as is. No deduplication,
  no subsampling, original row order.
  Output: train.tsv, dev.tsv, test.tsv ('label<TAB>text_a', UTF-8 without BOM, LF) and split_sizes.json.

Protocol 'edom18' (only for the comparison with the public 9-class result of edom18, ModernBERT-ja)
  All 2,500 official test rows. The released rounded 'Avg. Readers_<Emotion>' columns of the eight emotions
  (Sentiment excluded); all zero -> neutral; otherwise the largest, ties broken by the column order
  Joy, Sadness, Anticipation, Surprise, Anger, Fear, Disgust, Trust. Same text cleaning.
  Output: test.tsv.

Usage (standard library only):
  python tools/build_wrime.py                       writes data/wrime/
  python tools/build_wrime.py --protocol edom18     writes data/wrime_edom18/
  python tools/build_wrime.py --out DIR             writes to DIR instead
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'wrime' / 'source'
SOURCE_FILE = 'wrime-ver2.tsv'
DEFAULT_OUT = {'ours': ROOT / 'data' / 'wrime', 'edom18': ROOT / 'data' / 'wrime_edom18'}

REPO = 'https://github.com/ids-cv/wrime'
COMMIT = 'ac92deaff7845f31fdd46ef5d893300887006d13'
# Git blob SHA-1 of each release file on the master branch (GitHub contents API).
EXPECTED_BLOB = {
    'LICENSE': 'cfe676c541842bd2b766787b1f6983e063a1eee2',
    'README.en.md': 'a1371341a34d651f01c6550a03ed53b654004d99',
    'README.md': 'aee4490807527ffcc4660aaaec73b4471745fc6a',
    'wrime-ver2.tsv': 'd2e6080cea7e6a8c76148716c12521a276671a77',
}
SOURCE_SHA256 = '5be94b7b4b41382e01141007e28e7cfdfdb4a8ce2f53a69ba7bef618dbd90d50'
# SHA-256 of the files used for training and evaluation.
EXPECTED_OUTPUT = {
    'ours': {
        'train.tsv': 'd15cffc18b477f1f32e67e9440b704f933e039e9d0506d8a7f07d008a486ebc9',
        'dev.tsv': 'fb31b6995da7e03019573332d946da50b30aa44fac9eaa725648a8741c47ec87',
        'test.tsv': '33735ec8c03e9e51304dc4c97e887e02a39dfc9c26db7decc8eae2c091586e0c',
    },
    'edom18': {
        'test.tsv': 'ff31ed80b9628175529536ddb76450f3ae6792ec2a9f83e60aea32dc8fed6305',
    },
}
# Test-set counts reported for the edom18 rule; the rule must reproduce them.
EDOM18_REPORTED = {'neutral': 78, 'angry': 20, 'trust': 9}

EMO = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']   # file column order
NAMES = ['neutral', 'angry', 'happy', 'sad', 'fear', 'surprise', 'disgust', 'anticipation', 'trust']
SPLITS = ['train', 'dev', 'test']
KEPT = {'Anger': 1, 'Joy': 2, 'Sadness': 3, 'Fear': 4, 'Surprise': 5, 'Disgust': 6, 'Anticipation': 7, 'Trust': 8}
REASONS = ['empty text', 'tie']
ONE = Fraction(1)
WS = re.compile(r'[\t\r\n]')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path):
    data = path.read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def check_source():
    path = SOURCE / SOURCE_FILE
    if not path.is_file():
        raise FileNotFoundError(f'{path} not found; download it from {REPO}')
    for name, expected in EXPECTED_BLOB.items():
        if (SOURCE / name).is_file() and git_blob_sha1(SOURCE / name) != expected:
            raise RuntimeError(f'{name}: git blob SHA-1 differs from the release')
    if sha256(path) != SOURCE_SHA256:
        raise RuntimeError(f'{SOURCE_FILE}: SHA-256 differs from the release')
    return path


def read_source(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        recs = list(csv.reader(f, delimiter='\t', quotechar='"', doublequote=True, strict=True))
    header, body = recs[0], recs[1:]
    if any(len(r) != len(header) for r in body):
        raise ValueError('field-count mismatch in source TSV')
    needed = ['Sentence', 'Train/Dev/Test'] + [f'Reader{k}_{e}' for k in (1, 2, 3) for e in EMO] + \
             [f'Avg. Readers_{e}' for e in EMO]
    missing = [c for c in needed if c not in header]
    if missing:
        raise ValueError(f'missing columns: {missing}')
    return header, [dict(zip(header, r)) for r in body]


def reader_avgs(row):
    """Exact mean of the three reader intensities per emotion."""
    return {e: Fraction(sum(int(row[f'Reader{k}_{e}']) for k in (1, 2, 3)), 3) for e in EMO}


def assign(avg):
    """Return (label, None) or (None, reason)."""
    top = max(avg.values())
    if top < ONE:
        return 0, None
    winners = [e for e, v in avg.items() if v == top]
    if len(winners) > 1:
        return None, 'tie'
    return KEPT[winners[0]], None


def edom18_label(row):
    avg = [int(row[f'Avg. Readers_{e}']) for e in EMO]
    top = max(avg)
    return 0 if top == 0 else KEPT[EMO[avg.index(top)]]


def clean(text):
    return WS.sub(' ', text).strip()


def build_ours(rows):
    out = {s: [] for s in SPLITS}
    dropped = {s: Counter() for s in SPLITS}
    for row in rows:
        split = row['Train/Dev/Test']
        text = clean(row['Sentence'])
        if not text:
            dropped[split]['empty text'] += 1
            continue
        label, reason = assign(reader_avgs(row))
        if label is None:
            dropped[split][reason] += 1
            continue
        out[split].append((label, text))
    return out, dropped


def build_edom18(rows):
    test = [r for r in rows if r['Train/Dev/Test'] == 'test']
    if len(test) != 2500:
        raise RuntimeError(f'expected 2,500 official test rows, found {len(test)}')
    out = []
    for r in test:
        text = clean(r['Sentence'])
        if not text:
            raise RuntimeError('empty test text')
        out.append((edom18_label(r), text))
    counts = Counter(label for label, _ in out)
    got = {name: counts.get(NAMES.index(name), 0) for name in EDOM18_REPORTED}
    if got != EDOM18_REPORTED:
        raise RuntimeError(f'the rule does not reproduce the reported test counts: {got} vs {EDOM18_REPORTED}')
    return {'test': out}, {'test': Counter()}


def write_tsv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        f.write('label\ttext_a\n')
        for label, text in rows:
            if '\t' in text or '\n' in text or '\r' in text or not text:
                raise RuntimeError('bad text')
            f.write(f'{label}\t{text}\n')
    # Read back exactly as reasoning_training.read_rows does.
    with path.open(encoding='utf-8-sig', newline='') as f:
        rd = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        if rd.fieldnames != ['label', 'text_a']:
            raise RuntimeError('header mismatch')
        back = [(int(r['label']), r['text_a']) for r in rd]
    if back != rows:
        raise RuntimeError(f'round-trip mismatch in {path.name}')


def main(argv=None):
    ap = argparse.ArgumentParser(description='Recreate the WRIME ver2 splits from data/wrime/source/wrime-ver2.tsv.')
    ap.add_argument('--protocol', choices=['ours', 'edom18'], default='ours',
                    help="'ours' (default): train/dev/test used for training; 'edom18': alternative test set")
    ap.add_argument('--out', help='output folder (default: data/wrime or data/wrime_edom18)')
    args = ap.parse_args(argv)
    out_dir = Path(args.out) if args.out else DEFAULT_OUT[args.protocol]

    _, rows = read_source(check_source())
    if set(r['Train/Dev/Test'] for r in rows) != set(SPLITS):
        raise ValueError('unexpected values in the Train/Dev/Test column')
    splits, dropped = build_ours(rows) if args.protocol == 'ours' else build_edom18(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(protocol=args.protocol, source=f'{REPO} @ {COMMIT} / {SOURCE_FILE}', splits={})
    mismatched = []
    for split, pairs in splits.items():
        path = out_dir / f'{split}.tsv'
        write_tsv(path, pairs)
        digest = sha256(path)
        expected = EXPECTED_OUTPUT[args.protocol][path.name]
        if digest != expected:
            mismatched.append(path.name)
        counts = Counter(label for label, _ in pairs)
        summary['splits'][split] = dict(rows=len(pairs), class_counts=[counts.get(i, 0) for i in range(len(NAMES))],
                                        dropped={r: dropped[split][r] for r in REASONS},
                                        sha256=digest, matches_training_file=digest == expected)
    if args.protocol == 'ours':
        sizes = {s: len(splits[s]) for s in SPLITS}
        (out_dir / 'split_sizes.json').write_text(json.dumps(sizes), encoding='utf-8')
    summary['label_order'] = NAMES
    print(json.dumps(summary, indent=1))
    if mismatched:
        sys.exit(f'SHA-256 mismatch for {mismatched}; the files differ from the ones used in this work')
    print(f'Wrote {", ".join(f"{s}.tsv" for s in splits)} to {out_dir}; all match the training files.')


if __name__ == '__main__':
    main()
