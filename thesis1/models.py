"""Thesis 1 models: chinese-roberta-wwm-ext with five classification heads, trained on SMP2020-EWECT usual.

All five models fine-tune the whole encoder together with the head. They were trained on data/usual (same
train/dev/test files as the Qwen models) and predict the label ids of those files directly:
0 neutral, 1 angry, 2 happy, 3 sad, 4 fear, 5 surprise.

Input handling is the one used in training: the tokenizer of chinese-roberta-wwm-ext, [CLS] text [SEP], truncated
and always padded to 128 tokens. The padding length matters for the BiGRU heads, because the backward GRU direction
runs over the padded positions before it reaches the last real token.

    from models import THESIS1_MODELS, load_model
    model = load_model('roberta_bigru')                 # weights/thesis1/roberta_bigru.pth
    probs = model.predict_proba(['今天真开心'])         # numpy array, shape (1, 6)
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENCODER_SUBDIR = Path('models') / 'chinese-roberta-wwm-ext'
WEIGHTS_SUBDIR = Path('weights') / 'thesis1'
LABELS = ['neutral', 'angry', 'happy', 'sad', 'fear', 'surprise']
NUM_CLASSES = 6
MAX_LENGTH = 128
GRU_HIDDEN = 256

THESIS1_MODELS = [
    dict(key='roberta_textcnn', display_name='RoBERTa-TextCNN', file='roberta_textcnn.pth',
         description='Token states of the encoder, three 1-D convolutions (kernel 3/4/5, 100 filters each), '
                     'masked max pooling, linear layer.'),
    dict(key='roberta_bigru', display_name='RoBERTa-BiGRU', file='roberta_bigru.pth',
         description='Token states of the encoder, 2-layer bidirectional GRU (256 per direction), '
                     'GRU output at the last real token, linear layer.'),
    dict(key='roberta_linear', display_name='RoBERTa-Linear', file='roberta_linear.pth',
         description='[CLS] state of the encoder, linear layer.'),
    dict(key='roberta_bigru_attention', display_name='RoBERTa-BiGRU-Attention',
         file='roberta_bigru_attention.pth',
         description='Gated fusion of the [CLS] state and the projected BiGRU state, linear layer; '
                     'trained with cross-entropy only (lambda = 0).'),
    dict(key='roberta_bigru_attention_topo', display_name='RoBERTa-BiGRU-Attention + topological regulariser',
         file='roberta_bigru_attention_topo.pth',
         description='Same architecture as RoBERTa-BiGRU-Attention; trained with cross-entropy plus the '
                     'topological regulariser on the batch of predicted distributions (lambda = 0.15). '
                     'This is the full Thesis 1 model.'),
]
ARCHITECTURE = {'roberta_textcnn': 'textcnn', 'roberta_bigru': 'bigru', 'roberta_linear': 'linear',
         'roberta_bigru_attention': 'attention', 'roberta_bigru_attention_topo': 'attention'}


def model_info(key):
    for info in THESIS1_MODELS:
        if info['key'] == key:
            return info
    raise KeyError(f'unknown Thesis 1 model {key!r}; choose from {", ".join(m["key"] for m in THESIS1_MODELS)}')


def build_module(arch, encoder_dir, pretrained=False):
    """The torch module of one architecture. pretrained=True starts the encoder from the released
    chinese-roberta-wwm-ext weights (training); otherwise only its config is read (the fine-tuned state dict
    replaces every weight)."""
    import torch
    from torch import nn
    from transformers import AutoConfig, AutoModel

    def encoder():
        if pretrained:
            return AutoModel.from_pretrained(str(encoder_dir), local_files_only=True)
        return AutoModel.from_config(AutoConfig.from_pretrained(str(encoder_dir), local_files_only=True))

    def last_real_state(states, attention_mask):
        idx = attention_mask.sum(dim=1) - 1
        return states[torch.arange(states.size(0), device=states.device), idx]

    def bigru(size):
        return nn.GRU(input_size=size, hidden_size=GRU_HIDDEN, num_layers=2, batch_first=True,
                      dropout=0.3, bidirectional=True)

    class TextCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.roberta = encoder()
            size = self.roberta.config.hidden_size
            self.convs = nn.ModuleList([nn.Conv1d(size, 100, kernel_size=3, padding=1),
                                        nn.Conv1d(size, 100, kernel_size=4, padding=2),
                                        nn.Conv1d(size, 100, kernel_size=5, padding=2)])
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(300, NUM_CLASSES)

        def forward(self, input_ids, attention_mask):
            states = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            x = states.transpose(1, 2)
            seq_len = attention_mask.size(1)
            pad = ~attention_mask.unsqueeze(1).bool()
            pooled = []
            for conv in self.convs:
                out = torch.relu(conv(x))[:, :, :seq_len]      # kernel 4 gives one extra position
                pooled.append(out.masked_fill(pad, float('-inf')).max(dim=2).values)
            return self.fc(self.dropout(torch.cat(pooled, dim=1)))

    class BiGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.roberta = encoder()
            self.bigru = bigru(self.roberta.config.hidden_size)
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(2 * GRU_HIDDEN, NUM_CLASSES)

        def forward(self, input_ids, attention_mask):
            states = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            out, _ = self.bigru(states)
            return self.fc(self.dropout(last_real_state(out, attention_mask)))

    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.roberta = encoder()
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(self.roberta.config.hidden_size, NUM_CLASSES)

        def forward(self, input_ids, attention_mask):
            states = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            return self.fc(self.dropout(states[:, 0]))

    class BiGRUAttention(nn.Module):
        """Returns (logits, fused feature). The fused feature is a convex combination of the [CLS] state and the
        BiGRU state projected to the encoder width; the two weights come from small sigmoid scorers."""

        def __init__(self):
            super().__init__()
            self.roberta = encoder()
            size = self.roberta.config.hidden_size
            self.bigru = bigru(size)
            self.attention_cls = nn.Sequential(nn.Linear(size, size // 4), nn.ReLU(),
                                               nn.Linear(size // 4, 1), nn.Sigmoid())
            self.attention_gru = nn.Sequential(nn.Linear(2 * GRU_HIDDEN, GRU_HIDDEN), nn.ReLU(),
                                               nn.Linear(GRU_HIDDEN, 1), nn.Sigmoid())
            self.gru_to_embedding = nn.Linear(2 * GRU_HIDDEN, size)
            self.dropout = nn.Dropout(0.6)
            self.fc = nn.Linear(size, NUM_CLASSES)

        def forward(self, input_ids, attention_mask):
            states = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            cls = states[:, 0]
            out, _ = self.bigru(states)
            last = last_real_state(out, attention_mask)
            w_cls, w_gru = self.attention_cls(cls), self.attention_gru(last)
            total = w_cls + w_gru + 1e-8
            fused = (w_cls / total) * cls + (w_gru / total) * self.gru_to_embedding(last)
            return self.fc(self.dropout(fused)), fused

    return {'textcnn': TextCNN, 'bigru': BiGRU, 'linear': Linear, 'attention': BiGRUAttention}[arch]()


def logits_of(output):
    """The attention models return (logits, feature); the others return logits."""
    return output[0] if isinstance(output, tuple) else output


@lru_cache(maxsize=4)
def load_tokenizer(encoder_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(encoder_dir), local_files_only=True)


def encode(tokenizer, texts):
    """Token ids and attention mask, always padded to MAX_LENGTH (as in training)."""
    return tokenizer([str(t) for t in texts], max_length=MAX_LENGTH, padding='max_length', truncation=True,
                     return_tensors='pt')


class Thesis1Classifier:
    def __init__(self, key, module, tokenizer, device):
        self.key, self.info = key, model_info(key)
        self.display_name = self.info['display_name']
        self.module, self.tokenizer, self.device = module, tokenizer, device

    def predict_proba(self, texts, batch_size=32):
        """Class probabilities (softmax of the logits), shape (len(texts), 6), label order of LABELS."""
        import torch
        texts = list(texts)
        out = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = encode(self.tokenizer, texts[start:start + batch_size]).to(self.device)
                logits = logits_of(self.module(batch['input_ids'], batch['attention_mask'])).float()
                out.append(logits.softmax(-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, NUM_CLASSES), dtype=np.float32)

    def predict(self, texts, batch_size=32):
        return self.predict_proba(texts, batch_size).argmax(1)


def read_tsv(path):
    """(labels, texts) of a label<TAB>text_a file. Quotes are ordinary characters (QUOTE_NONE), as in training."""
    import csv
    with open(path, encoding='utf-8-sig', newline='') as handle:
        rows = [(int(r['label']), r['text_a']) for r in csv.DictReader(handle, delimiter='\t', quoting=csv.QUOTE_NONE)]
    if not rows or not all(label in range(NUM_CLASSES) and text.strip() for label, text in rows):
        raise ValueError(f'{path}: empty file, empty text or label outside 0-5')
    return [label for label, _ in rows], [text for _, text in rows]


def metrics(y, pred, k=NUM_CLASSES):
    """Accuracy, Macro-F1 (mean of the per-class F1), per-class precision / recall / F1, support, confusion."""
    y, pred = np.asarray(y), np.asarray(pred)
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(k), where=cm.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(k), where=(precision + recall) > 0)
    return dict(accuracy=float(tp.sum() / len(y)), macro_f1=float(f1.mean()), precision=precision.tolist(),
                recall=recall.tolist(), f1=f1.tolist(), support=cm.sum(1).tolist(), confusion=cm.tolist())


def pick_device(name='auto'):
    """'auto' -> cuda if available, else Apple-silicon mps if available, else cpu (same rule as tools/common.py)."""
    import os
    if name in ('auto', 'mps'):
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    import torch
    if name == 'auto':
        mps = getattr(torch.backends, 'mps', None)
        name = 'cuda' if torch.cuda.is_available() else 'mps' if mps is not None and mps.is_available() else 'cpu'
    return name


def load_model(key, device='cpu', root=None, encoder_dir=None, weights_dir=None):
    """Load one Thesis 1 model for inference (float32). device: cpu, cuda, mps or auto (see pick_device). Paths
    default to <root>/models/chinese-roberta-wwm-ext and <root>/weights/thesis1, with root = the package root."""
    device = pick_device(device)
    import torch
    info = model_info(key)
    root = Path(root) if root else ROOT
    encoder_dir = Path(encoder_dir) if encoder_dir else root / ENCODER_SUBDIR
    weights_dir = Path(weights_dir) if weights_dir else root / WEIGHTS_SUBDIR
    if not (encoder_dir / 'config.json').is_file():
        raise FileNotFoundError(f'encoder config not found in {encoder_dir}')
    weights = weights_dir / info['file']
    if not weights.is_file():
        raise FileNotFoundError(f'weights not found: {weights}')
    module = build_module(ARCHITECTURE[key], encoder_dir)
    module.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    module.to(device).eval()
    return Thesis1Classifier(key, module, load_tokenizer(encoder_dir), device)
