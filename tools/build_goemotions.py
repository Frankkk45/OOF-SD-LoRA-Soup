"""Rebuild the 7-class GoEmotions splits (Demszky et al., ACL 2020; Ekman classes + neutral) from the release.

Inputs, unchanged release files in data/goemotions/source/
(https://github.com/google-research/google-research/tree/master/goemotions):
  train.tsv, dev.tsv, test.tsv   official splits, no header: text <TAB> label ids <TAB> comment id
  emotions.txt                   the 28 label names in id order (neutral = 27)
  ekman_mapping.json             fine label -> Ekman category
  README.md, LICENSE             documentation and licence (Apache-2.0)

Mapping
  neutral -> 0; Ekman anger -> 1 (angry); joy -> 2 (happy); sadness -> 3 (sad); fear -> 4; surprise -> 5;
  disgust -> 6. A row is kept iff the set of mapped classes over all its labels has exactly one element.
  Dropped: 'neutral+emotion' (neutral plus an emotion), 'multi-class' (two or more emotion classes),
  'empty_text' (text empty after cleaning; checked after the label rule).
  Text: the release quotes comments that contain a double quote, so the files are parsed with CSV quoting;
  then every TAB, CR and LF becomes one space and the text is stripped. No subsampling, no deduplication,
  original row order.

Output: train.tsv, dev.tsv, test.tsv ('label<TAB>text_a', UTF-8 without BOM, LF) and split_sizes.json,
checked against the SHA-256 of the files used in training.

Usage (standard library only):
  python tools/build_goemotions.py              writes data/goemotions/
  python tools/build_goemotions.py --out DIR    writes to DIR instead
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'goemotions' / 'source'
DEFAULT_OUT = ROOT / 'data' / 'goemotions'

REPO = 'https://github.com/google-research/google-research'
SOURCE_COMMIT = '2adf640a14f11025ae5a9d0ec493b78530d276d3'
SPLITS = ['train', 'dev', 'test']
CLASS_NAMES = ['neutral', 'angry', 'happy', 'sad', 'fear', 'surprise', 'disgust']
EKMAN_TO_CLASS = {'anger': 1, 'disgust': 6, 'joy': 2, 'sadness': 3, 'fear': 4, 'surprise': 5}
EXPECTED_RAW_ROWS = {'train': 43410, 'dev': 5426, 'test': 5427}
CLEAN_RE = re.compile(r'[\t\r\n]')
# Git blob SHA-1 of the release files (GitHub contents API, goemotions/data on master).
GIT_BLOB_SHA1 = {
    'train.tsv': 'aa00d732d3b9e969d9fcbaafc2c5b094555746c0',
    'dev.tsv': '38c793645ca3a2e946d625c9722ca5386b0a6c7f',
    'test.tsv': '7635db4a49befb91e5bc0dbedde095bb2d8593ae',
    'emotions.txt': '0d79bc030e766bc9dac1f4482bfbd864556c284a',
    'ekman_mapping.json': 'dd13b0702f20993f0f4aed84fc87a124c7a783ba',
}
# SHA-256 of the files used for training and evaluation.
EXPECTED_OUTPUT = {
    'train.tsv': '370f88be0712627acc93a736b12b5ef35159472608552bebb4f2b2355f89cdfb',
    'dev.tsv': 'a621c77f323cbd57a85d1471aa179c29b6f371a7d6b0d254ed5e82972a399ad2',
    'test.tsv': 'b636dfaab169ba937cb44bebd1ab3785191f2fff9525e5731426932883ca0f37',
}


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path):
    data = path.read_bytes()
    return hashlib.sha1(b'blob %d\x00' % len(data) + data).hexdigest()


def load_label_maps(source):
    names = (source / 'emotions.txt').read_text(encoding='utf-8').split()
    check(len(names) == 28 and names[27] == 'neutral', 'emotions.txt must list 28 names ending with neutral')
    with open(source / 'ekman_mapping.json', encoding='utf-8') as handle:
        ekman = json.load(handle)
    check(set(ekman) == set(EKMAN_TO_CLASS), f'unexpected Ekman categories: {sorted(ekman)}')
    fine_to_ekman = {}
    for category, members in ekman.items():
        for member in members:
            check(member in names and member not in fine_to_ekman, f'bad Ekman member: {member}')
            fine_to_ekman[member] = category
    check(set(fine_to_ekman) == set(names[:27]), 'the Ekman mapping must cover the 27 emotion labels')
    id_to_class = {27: 0}
    for idx, name in enumerate(names[:27]):
        id_to_class[idx] = EKMAN_TO_CLASS[fine_to_ekman[name]]
    return names, ekman, id_to_class


def read_raw_split(path):
    data = path.read_bytes()
    check(not data.startswith(b'\xef\xbb\xbf'), f'{path.name}: unexpected BOM')
    text = data.decode('utf-8')
    records = list(csv.reader(io.StringIO(text, newline=''), delimiter='\t', quotechar='"', doublequote=True, strict=True))
    rows = []
    for line_no, fields in enumerate(records):
        if len(fields) != 3:
            raise ValueError(f'{path.name}:{line_no + 1}: expected 3 tab-separated fields, got {len(fields)}')
        raw_text, label_field, comment_id = fields
        label_ids = [int(x) for x in label_field.split(',')]
        check(label_ids and all(0 <= x <= 27 for x in label_ids), f'{path.name}:{line_no + 1}: bad label ids')
        rows.append(dict(raw_text=raw_text, label_ids=label_ids, comment_id=comment_id))
    return rows


def convert_split(rows, id_to_class, disgust_id, anger_group_ids):
    kept, dropped = [], collections.Counter()
    for row in rows:
        classes = {id_to_class[x] for x in row['label_ids']}
        if len(classes) != 1:
            dropped['neutral+emotion' if 0 in classes else 'multi-class'] += 1
            continue
        cleaned = CLEAN_RE.sub(' ', row['raw_text']).strip()
        if not cleaned:
            dropped['empty_text'] += 1
            continue
        label = next(iter(classes))
        check((label != 1 or set(row['label_ids']) <= anger_group_ids)
              and (label != 6 or set(row['label_ids']) == {disgust_id}), 'inconsistent Ekman mapping')
        kept.append((label, cleaned))
    return kept, dropped


def write_tsv(path, kept):
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        handle.write('label\ttext_a\n')
        for label, text in kept:
            check('\t' not in text and '\n' not in text and '\r' not in text and text, 'bad text')
            handle.write(f'{label}\t{text}\n')


def verify_tsv(path, kept):
    """Read back exactly as reasoning_training.read_rows does and compare."""
    with open(path, encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t', quoting=csv.QUOTE_NONE)
        check(reader.fieldnames == ['label', 'text_a'], f'header mismatch in {path.name}')
        back = []
        for index, row in enumerate(reader):
            check(None not in row and row.get('text_a') is not None, f'malformed row {index}')
            label = int(row['label'])
            check(label in range(len(CLASS_NAMES)) and row['text_a'].strip(), f'invalid row {index}')
            back.append((label, row['text_a']))
    check(back == kept, f'round-trip mismatch in {path.name}')


def main(argv=None):
    ap = argparse.ArgumentParser(description='Rebuild the 7-class GoEmotions splits from data/goemotions/source/.')
    ap.add_argument('--source', default=str(SOURCE), help='folder with the release files (default: data/goemotions/source)')
    ap.add_argument('--out', default=str(DEFAULT_OUT), help='output folder (default: data/goemotions)')
    args = ap.parse_args(argv)
    source, out_dir = Path(args.source), Path(args.out)

    for name, expected in GIT_BLOB_SHA1.items():
        check((source / name).is_file(), f'{source / name} not found; download it from {REPO}')
        check(git_blob_sha1(source / name) == expected, f'{name} differs from the release file')
    names, ekman, id_to_class = load_label_maps(source)
    disgust_id = names.index('disgust')
    anger_group_ids = {names.index(n) for n in ekman['anger']}

    out_dir.mkdir(parents=True, exist_ok=True)
    summary, sizes, mismatched = {}, {}, []
    for split in SPLITS:
        rows = read_raw_split(source / f'{split}.tsv')
        check(len(rows) == EXPECTED_RAW_ROWS[split], f'{split}: expected {EXPECTED_RAW_ROWS[split]} rows, found {len(rows)}')
        kept, dropped = convert_split(rows, id_to_class, disgust_id, anger_group_ids)
        out = out_dir / f'{split}.tsv'
        write_tsv(out, kept)
        verify_tsv(out, kept)
        digest = sha256(out)
        if digest != EXPECTED_OUTPUT[out.name]:
            mismatched.append(out.name)
        counts = collections.Counter(label for label, _ in kept)
        sizes[split] = len(kept)
        summary[split] = dict(raw_rows=len(rows), kept_rows=len(kept),
                              class_counts=[counts.get(i, 0) for i in range(len(CLASS_NAMES))],
                              dropped={k: dropped.get(k, 0) for k in ('neutral+emotion', 'multi-class', 'empty_text')},
                              sha256=digest, matches_training_file=digest == EXPECTED_OUTPUT[out.name])
    (out_dir / 'split_sizes.json').write_text(json.dumps(sizes), encoding='utf-8')
    print(json.dumps(dict(source=f'{REPO} @ {SOURCE_COMMIT} / goemotions/data', label_order=CLASS_NAMES,
                          splits=summary), indent=1))
    if mismatched:
        sys.exit(f'SHA-256 mismatch for {mismatched}; the files differ from the ones used in this work')
    print(f'Wrote train.tsv, dev.tsv, test.tsv and split_sizes.json to {out_dir}; all match the training files.')


if __name__ == '__main__':
    main()
