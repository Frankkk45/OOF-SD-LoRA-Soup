"""Train one Thesis 1 model on data/usual (script version of the two Thesis 1 training notebooks).

  python thesis1/train.py --model roberta_bigru --out runs/thesis1_bigru
  python thesis1/train.py --model roberta_bigru_attention_topo --out runs/thesis1_topo            # lambda 0.15
  python thesis1/train.py --model roberta_bigru_attention --lambda 0.0 --out runs/thesis1_attention

Recipe (identical for all five models):
- encoder chinese-roberta-wwm-ext, fine-tuned end to end; inputs truncated and padded to 128 tokens;
- data read with QUOTE_NONE (27,766 / 2,000 / 5,000 rows); batch 32, shuffled;
- AdamW (torch defaults otherwise), learning rate 2e-5 for the encoder and 4e-4 for every other layer
  (convolutions, GRU, attention scorers, projection, classifier); the optimiser is checked to hold every
  trainable parameter exactly once;
- float16 autocast with a gradient scaler on CUDA;
- at most 5 epochs; after each epoch the dev Macro-F1 is computed, the best epoch is kept, and training stops
  after 3 epochs without improvement; the kept weights are scored once on test;
- loss = cross-entropy, plus lambda * topological regulariser for the attention models (lambda 0.15 for
  roberta_bigru_attention_topo, 0 for roberta_bigru_attention). The regulariser is computed on the batch of
  predicted distributions softmax(logits): pairwise cosine distances, variance of each row, mean over rows.
- seed 42 is set right before the model is built (as in the corrected ablation run). The three baselines of
  Thesis 1 were trained one after another in a single process after one seed call, and GPU training is not
  bit-deterministic, so a retrained model will differ slightly from the shipped weights.

Writes <out>/<model>.pth (state dict of the selected epoch), <out>/history.json and <out>/results.json.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import (ARCHITECTURE, ENCODER_SUBDIR, ROOT, THESIS1_MODELS, build_module, encode,  # noqa: E402
                    load_tokenizer, logits_of, metrics, read_tsv)

KEYS = [m['key'] for m in THESIS1_MODELS]
SEED = 42
BATCH_SIZE = 32
EPOCHS = 5
PATIENCE = 3
ENCODER_LR = 2e-5
HEAD_LR = 4e-4
DEFAULT_LAMBDA = {'roberta_bigru_attention_topo': 0.15}
HEAD_MODULES = ('convs', 'bigru', 'attention_cls', 'attention_gru', 'gru_to_embedding', 'fc')


def topological_regulariser(features):
    """Mean over the batch of the variance of each row of the pairwise cosine-distance matrix."""
    import torch
    import torch.nn.functional as F
    z = F.normalize(features, p=2, dim=1)
    dist = torch.clamp(1.0 - z @ z.t(), min=0.0)
    return ((dist - dist.mean(dim=1, keepdim=True)) ** 2).mean(dim=1).mean()


def make_optimizer(module):
    import torch
    groups = [{'params': list(module.roberta.parameters()), 'lr': ENCODER_LR}]
    for name in HEAD_MODULES:
        if hasattr(module, name):
            groups.append({'params': list(getattr(module, name).parameters()), 'lr': HEAD_LR})
    optimizer = torch.optim.AdamW(groups)
    trainable = [p for p in module.parameters() if p.requires_grad]
    held = [p for g in optimizer.param_groups for p in g['params']]
    if {id(p) for p in trainable} != {id(p) for p in held} or len(held) != len(trainable):
        raise RuntimeError('the optimiser must hold every trainable parameter exactly once')
    return optimizer


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model', required=True, choices=KEYS)
    parser.add_argument('--lambda', dest='lam', type=float,
                        help='weight of the topological regulariser (attention models only; default 0.15 for '
                             'roberta_bigru_attention_topo, otherwise 0)')
    parser.add_argument('--out', type=Path, required=True, help='output folder')
    parser.add_argument('--root', type=Path, help='package root (default: the folder above thesis1/)')
    parser.add_argument('--data-dir', type=Path, help='default: <root>/data/usual')
    parser.add_argument('--encoder-dir', type=Path, help='default: <root>/models/chinese-roberta-wwm-ext')
    parser.add_argument('--device', default='auto', help='auto, cuda or cpu')
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    root = args.root or ROOT
    data_dir = args.data_dir or root / 'data' / 'usual'
    encoder_dir = args.encoder_dir or root / ENCODER_SUBDIR
    arch = ARCHITECTURE[args.model]
    lam = DEFAULT_LAMBDA.get(args.model, 0.0) if args.lam is None else args.lam
    if lam and arch != 'attention':
        raise SystemExit('--lambda applies only to roberta_bigru_attention and roberta_bigru_attention_topo')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    use_amp = device.startswith('cuda')
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(encoder_dir)
    splits = {}
    for split in ('train', 'dev', 'test'):
        labels, texts = read_tsv(data_dir / f'{split}.tsv')
        enc = encode(tokenizer, texts)
        splits[split] = TensorDataset(enc['input_ids'], enc['attention_mask'], torch.tensor(labels))
        print(f'{split}: {len(labels)} rows', flush=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    module = build_module(arch, encoder_dir, pretrained=True).to(device)
    optimizer = make_optimizer(module)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(splits['train'], batch_size=BATCH_SIZE, shuffle=True)

    def autocast():
        return torch.autocast('cuda', dtype=torch.float16, enabled=use_amp)

    def score(split):
        module.eval()
        preds, golds, loss_sum, batches = [], [], 0.0, 0
        with torch.no_grad():
            for ids, mask, y in DataLoader(splits[split], batch_size=BATCH_SIZE, shuffle=False):
                ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                with autocast():
                    logits = logits_of(module(ids, mask))
                    loss_sum += criterion(logits, y).item()
                batches += 1
                preds.append(logits.argmax(1).cpu())
                golds.append(y.cpu())
        m = metrics(torch.cat(golds).numpy(), torch.cat(preds).numpy())
        m['loss'] = loss_sum / batches
        return m

    print(f'model {args.model}, lambda {lam}, device {device}', flush=True)
    history, best_f1, best_state, best_epoch, stale = [], -1.0, None, None, 0
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        module.train()
        total = 0.0
        for ids, mask, y in train_loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                logits = logits_of(module(ids, mask))
                loss = criterion(logits, y)
                if lam > 0:
                    loss = loss + lam * topological_regulariser(torch.softmax(logits, dim=1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        dev = score('dev')
        record = dict(epoch=epoch, train_loss=total / len(train_loader), dev_loss=dev['loss'],
                      dev_accuracy=dev['accuracy'], dev_macro_f1=dev['macro_f1'], seconds=time.time() - start)
        history.append(record)
        print(f'epoch {epoch}/{EPOCHS}  train loss {record["train_loss"]:.4f}  dev loss {dev["loss"]:.4f}  '
              f'dev acc {dev["accuracy"]:.4f}  dev macro-F1 {dev["macro_f1"]:.4f}', flush=True)
        (args.out / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        if dev['macro_f1'] > best_f1:
            best_f1, best_epoch, stale = dev['macro_f1'], epoch, 0
            best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f'early stop after {PATIENCE} epochs without improvement', flush=True)
                break

    module.load_state_dict(best_state)
    test = score('test')
    torch.save({k: v.cpu() for k, v in best_state.items()}, args.out / f'{args.model}.pth')
    results = dict(model=args.model, topo_lambda=lam, seed=SEED, best_epoch=best_epoch, best_dev_macro_f1=best_f1,
                   test=dict(accuracy=test['accuracy'], macro_f1=test['macro_f1'], f1=test['f1'],
                             confusion=test['confusion']))
    (args.out / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'test (epoch {best_epoch}): accuracy {test["accuracy"]:.4f}  macro-F1 {test["macro_f1"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
