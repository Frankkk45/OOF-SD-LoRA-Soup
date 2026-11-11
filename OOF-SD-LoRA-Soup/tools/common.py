"""Shared helpers for the package tools: datasets, prompts, data reading, model lookup and metrics (no torch import).

Every model in the package was trained with one fixed prompt per dataset (system turn = instruction, user turn =
prefix + text cut to 128 tokens, thinking off) and is scored by a softmax over its answer-word tokens. The strings
below are byte-identical to the training configs; the answer words are Chinese for every dataset because the class
list is stated in Chinese in the instruction.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / 'models' / 'Qwen3.5-4B'
MAX_TEXT_TOKENS = 128

LABELS6 = ['neutral', 'angry', 'happy', 'sad', 'fear', 'surprise']
LABELS7 = LABELS6 + ['disgust']
LABELS9 = LABELS7 + ['anticipation', 'trust']
WORDS6 = ['中性', '愤怒', '喜悦', '悲伤', '恐惧', '惊讶']
WORDS7 = WORDS6 + ['厌恶']
WORDS9 = WORDS7 + ['期待', '信任']

WEIBO_INSTRUCTION = '判断下面微博的主要情绪，只输出一个情绪类别。类别：中性、愤怒、喜悦、悲伤、恐惧、惊讶。'
WEIBO_PREFIX = '微博文本：\n'
GOEMOTIONS_INSTRUCTION = '判断下面文本的主要情绪，只输出一个情绪类别。类别：中性、愤怒、喜悦、悲伤、恐惧、惊讶、厌恶。'
WRIME_INSTRUCTION = '判断下面文本的主要情绪，只输出一个情绪类别。类别：中性、愤怒、喜悦、悲伤、恐惧、惊讶、厌恶、期待、信任。'
TEXT_PREFIX = '文本：\n'

# Kept for code that only handles the two SMP2020-EWECT domains.
LABELS, ANSWER_WORDS, INSTRUCTION = LABELS6, WORDS6, WEIBO_INSTRUCTION

MEMBERS = ['SD_s42', 'SD_s43', 'SD_s44', 'SDfold1', 'SDfold2', 'SDgen2']
BASELINES = ['A_s42', 'A_s43', 'A_s44']
TEACHERS = ['Afold1', 'Afold2']
SOUPS = {'soup': 'sd_uniform6_soup', 'a_soup': 'a_uniform3_soup'}
MODEL_NAMES = list(SOUPS) + BASELINES + MEMBERS + TEACHERS
MODEL_HELP = 'soup (final model), a_soup, A_s42/43/44, SD_s42/43/44, SDfold1/2, SDgen2, Afold1/2, or an adapter folder'


class Domain:
    """One evaluation setting: which data, which weights, and the prompt the weights were trained with."""

    def __init__(self, key, title, labels, words, instruction, prefix, data=None, train_data=None, weights=None,
                 splits=('train', 'dev', 'test'), sizes=None, build_hint=None, note=''):
        self.key, self.title, self.labels, self.words = key, title, list(labels), list(words)
        self.instruction, self.prefix = instruction, prefix
        self.data = data or key                      # folder under data/ for dev and test
        self.train_data = train_data or self.data    # folder under data/ for train
        self.weights = weights or key                # folder under weights/
        self.splits, self.sizes, self.build_hint, self.note = tuple(splits), sizes, build_hint, note
        assert len(self.labels) == len(self.words) == len(set(self.words))

    @property
    def k(self):
        return len(self.labels)

    def folder(self, part):
        name = self.train_data if part == 'train' else self.data
        return DATA_OVERRIDE.get(name, ROOT / 'data' / name)


# Data folder name -> another folder. Used by check_package.py to read WRIME splits rebuilt in a temporary folder.
DATA_OVERRIDE = {}


DOMAINS = {
    'usual': Domain('usual', 'SMP2020-EWECT general Weibo (Chinese, 6 classes)', LABELS6, WORDS6, WEIBO_INSTRUCTION,
                    WEIBO_PREFIX),
    'virus': Domain('virus', 'SMP2020-EWECT COVID-19 Weibo (Chinese, 6 classes)', LABELS6, WORDS6, WEIBO_INSTRUCTION,
                    WEIBO_PREFIX),
    'goemotions': Domain('goemotions', 'GoEmotions (English, 7 classes)', LABELS7, WORDS7, GOEMOTIONS_INSTRUCTION,
                         TEXT_PREFIX),
    'wrime': Domain('wrime', 'WRIME ver2 (Japanese, 9 classes)', LABELS9, WORDS9, WRIME_INSTRUCTION, TEXT_PREFIX,
                    build_hint='python tools/build_wrime.py'),
    # The WRIME models scored on the official WRIME test split relabelled with the edom18 public 9-class rule.
    'wrime_edom18': Domain('wrime_edom18', 'WRIME ver2 test split, edom18 relabelling (Japanese, 9 classes)', LABELS9,
                           WORDS9, WRIME_INSTRUCTION, TEXT_PREFIX, weights='wrime', splits=('test',),
                           sizes={'test': 2500}, build_hint='python tools/build_wrime.py --protocol edom18',
                           note='weights/wrime, scored on data/wrime_edom18 (test only)'),
}
DATASETS = ('usual', 'virus', 'goemotions', 'wrime')      # the four datasets with their own weights and splits


DEVICE_CHOICES = ['auto', 'cuda', 'mps', 'cpu']


def pick_device(name='auto'):
    """'auto' -> cuda if available, else Apple-silicon mps if available, else cpu. On mps, unsupported operators fall
    back to the CPU (PYTORCH_ENABLE_MPS_FALLBACK=1)."""
    import os
    if name == 'mps' or name == 'auto':
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    import torch
    if name == 'auto':
        mps = getattr(torch.backends, 'mps', None)
        name = 'cuda' if torch.cuda.is_available() else 'mps' if mps is not None and mps.is_available() else 'cpu'
    return name


def pick_dtype(device, cpu='bfloat16'):
    """torch dtype name for a device: cuda -> bfloat16; mps -> bfloat16 if a small bf16 matmul works there, else
    float32 (float16 is not used: Qwen activations can exceed its range); cpu -> `cpu`."""
    import torch
    if str(device).startswith('cuda'):
        return 'bfloat16'
    if str(device).startswith('mps'):
        try:
            x = torch.ones(4, 4, dtype=torch.bfloat16, device=device)
            if torch.isfinite((x @ x).float().cpu()).all():
                return 'bfloat16'
        except (RuntimeError, TypeError):
            pass
        return 'float32'
    return cpu


def domain(key):
    if key not in DOMAINS:
        raise SystemExit(f'unknown domain {key!r}: use one of {", ".join(DOMAINS)}')
    return DOMAINS[key]


def model_path(dom, model):
    """Adapter folder of a model name (see MODEL_HELP) for a domain, or `model` itself if it is an adapter folder."""
    d = domain(dom) if isinstance(dom, str) else dom
    base = ROOT / 'weights' / d.weights
    if model in SOUPS:
        path = base / SOUPS[model]
    elif model in BASELINES:
        path = base / 'baseline' / model
    elif model in MEMBERS:
        path = base / 'members' / model
    elif model in TEACHERS:
        path = base / 'teachers' / model
    else:
        path = Path(model)
    if not (path / 'adapter_config.json').is_file():
        raise SystemExit(f'no adapter at {path}: use {MODEL_HELP}')
    return path


def split_sizes(dom, part):
    d = domain(dom) if isinstance(dom, str) else dom
    path = d.folder(part) / 'split_sizes.json'
    if path.is_file():
        return json.loads(path.read_text(encoding='utf-8'))[part]
    if d.sizes and part in d.sizes:
        return d.sizes[part]
    raise SystemExit(f'{rel(path)} not found')


def load_split(dom, part):
    """Rows of a split: dict(row_index, label, text). Quotes are ordinary characters (QUOTE_NONE), as in training."""
    d = domain(dom) if isinstance(dom, str) else dom
    if part not in d.splits:
        raise SystemExit(f'{d.key} has no {part} split (available: {", ".join(d.splits)})')
    path = d.folder(part) / f'{part}.tsv'
    if not path.is_file():
        hint = f'; run  {d.build_hint}  first' if d.build_hint else ''
        raise SystemExit(f'{rel(path)} not found{hint}')
    with open(path, encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t', quoting=csv.QUOTE_NONE)
        if reader.fieldnames != ['label', 'text_a']:
            raise RuntimeError(f'{path.name}: unexpected columns {reader.fieldnames}')
        rows = [dict(row_index=i, label=int(r['label']), text=r['text_a']) for i, r in enumerate(reader)]
    if len(rows) != split_sizes(d, part) or not all(r['label'] in range(d.k) and r['text'].strip() for r in rows):
        raise RuntimeError(f'{rel(path)} does not have the expected rows')
    return rows


def rel(path):
    """Package-relative POSIX path when inside the package, else the path itself."""
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def prediction_rows(rows, P):
    """Per-row prediction records in the format of reference/predictions (no text)."""
    return [dict(row_index=r['row_index'], gold=r['label'], predicted=int(p.argmax()), probabilities=[float(v) for v in p])
            for r, p in zip(rows, P)]


def metrics(y, pred, k):
    """Accuracy, Macro-F1 (unweighted mean of the per-class F1, the SMP2020-EWECT metric), support-weighted F1,
    per-class precision / recall / F1, support and confusion matrix (rows = gold)."""
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(k), where=cm.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(k), where=(precision + recall) > 0)
    support = cm.sum(1)
    return dict(accuracy=float(tp.sum() / len(y)), macro_f1=float(f1.mean()),
                weighted_f1=float((f1 * support).sum() / support.sum()), precision=precision.tolist(),
                recall=recall.tolist(), f1=f1.tolist(), support=support.tolist(), confusion=cm.tolist())


def print_metrics(m, title, dom):
    d = domain(dom) if isinstance(dom, str) else dom
    print(f'\n{title}')
    print(f'  Accuracy {100 * m["accuracy"]:.2f}   Macro-F1 {100 * m["macro_f1"]:.2f}   Weighted-F1 {100 * m["weighted_f1"]:.2f}')
    print('  class           precision  recall      F1  support')
    for i, name in enumerate(d.labels):
        print(f'  {name:14s} {100 * m["precision"][i]:9.2f} {100 * m["recall"][i]:7.2f} {100 * m["f1"][i]:7.2f} {m["support"][i]:8d}')
    print('  confusion matrix (rows = gold, columns = predicted, class order as above)')
    for name, row in zip(d.labels, m['confusion']):
        print(f'  {name:14s}' + ''.join(f'{v:6d}' for v in row))
