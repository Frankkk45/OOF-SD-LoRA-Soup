"""LoRA fine-tuning of Qwen3.5-4B for single-label emotion classification.

All package configs use mode 'post_cot' with use_rationales=false: the class is
read from one forward pass at the last prompt position (thinking off) as the
argmax over the answer-word tokens. The training loss is the cross-entropy of
the gold answer token or, with label_targets_file, the soft-target loss used for
out-of-fold self-distillation. The modes 'pre_cot' and 'think' (rationale
training) come from earlier experiments and need evaluator modules that are not
part of this package.

The supplied train/dev/test splits are used as they are. The adapter with the
best dev macro-F1 is selected and only that adapter is evaluated on test.
Interrupted runs resume from runs/<run_name>/resume.pt.

Requires ROOT/models/Qwen3.5-4B, ROOT/model_manifest.json and a BF16-capable
CUDA GPU. Usage: python reasoning_training.py --config configs/<name>.json,
or run(ROOT, config) from Python.
"""
from pathlib import Path
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import inspect
import random
import time
from generation_guard import replace_with_retry


LABELS = ['neutral', 'angry', 'happy', 'sad', 'fear', 'surprise']
TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'in_proj_qkv', 'in_proj_z',
           'in_proj_b', 'in_proj_a', 'out_proj', 'gate_proj', 'up_proj', 'down_proj']
INSTRUCTION = '判断下面微博的主要情绪，只输出一个数字。标签：0=中性，1=愤怒，2=喜悦，3=悲伤，4=恐惧，5=惊讶。'
# label_format='word': answer with the Chinese class name; each is one Qwen3.5 token.
LABEL_WORDS = ['中性', '愤怒', '喜悦', '悲伤', '恐惧', '惊讶']
WORD_INSTRUCTION = '判断下面微博的主要情绪，只输出一个情绪类别。类别：中性、愤怒、喜悦、悲伤、恐惧、惊讶。'
B_INSTRUCTION = '判断下面微博的主要情绪。先给出简短的原文依据与情绪说明，最后单独一行输出“标签：<数字>”。标签：0=中性，1=愤怒，2=喜悦，3=悲伤，4=恐惧，5=惊讶。证据不足时简短说明，不要编造背景。'
# Native Qwen3.5 thinking turn after a prompt that ends in '<think>\n'.
THINK_CLOSE = '\n</think>\n\n'
THINK_END = '<|im_end|>'
MODES = ('post_cot', 'pre_cot', 'think')
GENERATIVE_MODES = ('pre_cot', 'think')
DEFAULTS = dict(seed=42, epochs=3, patience=1, batch_size=4, gradient_accumulation=4,
                eval_batch_size=8, learning_rate=2e-4, weight_decay=0.01,
                rank=8, lora_alpha=16, dropout=0.05, max_text_tokens=128,
                max_prompt_tokens=256, max_grad_norm=1.0,
                checkpoint_steps=100, progress_batches=50, run_name='qwen35_post_cot',
                mode='post_cot', rationale_weight=1.0, format_weight=1.0,
                head_chunk_tokens=32, max_rationale_tokens=None,
                rationale_file='rationales/GT_train.jsonl', generation_max_new_tokens=None,
                thinking_budget=512, initial_dev_eval=True, label_format='digit', use_rationales=True,
                train_cleaning=None,
                # Optional training variants. Every default keeps the original behaviour.
                use_dora=False, use_rslora=False, loraplus_ratio=None, neftune_alpha=None, rdrop_alpha=None,
                label_targets_file=None, instruction_override=None, demo_file=None,
                # Optional verbalizer: one answer string per class in label order, each a distinct single token.
                answer_words=None,
                # Optional dataset folder under the experiment root with train/dev/test.tsv ('label', 'text_a')
                # and split_sizes.json; text_prefix replaces the user-turn prefix. None keeps the default
                # usual-domain folder and the default Weibo text prefix set in run().
                data_dir=None, text_prefix=None,
                # Optional class list: names of K classes in label order (TSV labels 0..K-1);
                # needs data_dir and answer_words of the same length. None keeps the six classes of LABELS.
                label_names=None)


def label_texts(label_format):
    if label_format == 'digit':
        return [str(i) for i in range(6)]
    if label_format == 'word':
        return list(LABEL_WORDS)
    raise ValueError("label_format must be 'digit' or 'word'")


def instruction_for(mode, label_format):
    if mode == 'pre_cot':
        if label_format != 'digit':
            raise ValueError('pre_cot keeps its digit marker format')
        return B_INSTRUCTION
    return WORD_INSTRUCTION if label_format == 'word' else INSTRUCTION


def json_friendly(value):
    if isinstance(value, dict):
        return {key: json_friendly(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [json_friendly(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [json_friendly(item) for item in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(json_friendly(value), handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_retry(temporary, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path, expected_size, num_labels=6):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t', quoting=csv.QUOTE_NONE)
        if reader.fieldnames != ['label', 'text_a']:
            raise ValueError(f'Unexpected columns in {path}: {reader.fieldnames}')
        rows = []
        for index, row in enumerate(reader):
            if None in row or row.get('text_a') is None:
                raise ValueError(f'Malformed row {index}: {path}')
            label = int(row['label'])
            if label not in range(num_labels) or not row['text_a'].strip():
                raise ValueError(f'Invalid label/text at row {index}: {path}')
            rows.append(dict(row_index=index, text=row['text_a'], label=label))
    if len(rows) != expected_size:
        raise ValueError(f'{path}: expected {expected_size} rows, found {len(rows)}')
    return rows


def apply_cleaning(rows, spec, num_labels=6):
    """Drop or relabel training rows by original row_index; row_index is kept so rationales still align."""
    drop = set(spec['drop'])
    relabel = {int(index): label for index, label in spec['relabel'].items()}
    if len(drop) != len(spec['drop']) or not drop.isdisjoint(relabel) or \
            not all(type(index) is int and 0 <= index < len(rows) for index in drop | set(relabel)):
        raise ValueError('Cleaning spec indices must be distinct, valid training rows')
    cleaned = []
    for row in rows:
        if row['row_index'] in drop:
            continue
        if row['row_index'] in relabel:
            label = relabel[row['row_index']]
            if label not in range(num_labels) or label == row['label']:
                raise ValueError(f'Invalid relabel at train row {row["row_index"]}')
            row = dict(row, label=label)
        cleaned.append(row)
    if len(cleaned) != spec['result_rows']:
        raise ValueError(f'Cleaning produced {len(cleaned)} rows; the spec expects {spec["result_rows"]}')
    return cleaned


def duplicate_summary(left, right):
    left_texts = {row['text'] for row in left}
    right_texts = {row['text'] for row in right}
    shared = left_texts & right_texts
    return dict(shared_unique_texts=len(shared),
                left_rows_with_shared_text=sum(row['text'] in shared for row in left),
                right_rows_with_shared_text=sum(row['text'] in shared for row in right))


def epoch_batches(size, batch_size, seed, epoch):
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    return [order[start:start + batch_size] for start in range(0, size, batch_size)]


def load_rationales(path, signature):
    rows = []
    with Path(path).open(encoding='utf-8') as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            if row.get('row_index') != index or row.get('signature') != signature:
                raise RuntimeError(f'Rationale order/signature mismatch at row {index}')
            if not row.get('retain_label_supervision'):
                raise RuntimeError('Every rationale row must retain label supervision')
            if not row.get('passes_heuristic_checks') and row.get('rationale_for_training'):
                raise RuntimeError('Rejected rationale unexpectedly contains a training target')
            rows.append(row)
    if len(rows) != 27766:
        raise RuntimeError(f'Expected 27766 rationale records, found {len(rows)}')
    return rows


def context_capacity(config):
    text_config = config.get('text_config', config) if isinstance(config, dict) else getattr(config, 'text_config', config)
    capacity = text_config.get('max_position_embeddings') if isinstance(text_config, dict) else getattr(text_config, 'max_position_embeddings', None)
    if type(capacity) is not int or capacity < 1:
        raise ValueError('Model config must declare a positive max_position_embeddings')
    return capacity


def training_sequence(tokenizer, prompt_ids, label, reason, mode, code_ids, model_context_capacity=None, label_text=None):
    """Return causal inputs and separately masked label/reason/format targets.

    Position p predicts the token at p+1. C label's prediction position is the
    final prompt position; its hidden state cannot attend to the following GT.
    For B/C, EOS is not supervised; C newline is not supervised; B marker has
    its own loss. For think, the rationale fills the native thinking block and
    '</think>', the separator and the final <|im_end|> are format targets.
    label_text is the answer string (a digit by default, or a one-token class word).
    """
    label_text = str(label) if label_text is None else label_text
    width = len(label_text)
    if mode == 'post_cot':
        target_text = label_text + ('\n' + reason if reason else '')
        label_start, reason_start, reason_end = 0, width + 1, len(target_text)
        marker_start = marker_end = -1
    elif mode == 'think':
        target_text = reason + THINK_CLOSE + label_text + THINK_END
        reason_start, reason_end = 0, len(reason)
        label_start = len(reason) + len(THINK_CLOSE)
        marker_start, marker_end = len(reason), len(target_text)
    else:
        marker = '\n标签：' if reason else '标签：'
        target_text = reason + marker + label_text
        reason_start, reason_end = 0, len(reason)
        marker_start, marker_end = len(reason), len(reason) + len(marker)
        label_start = len(target_text) - width
    encoded = tokenizer(target_text, add_special_tokens=False, return_offsets_mapping=True)
    target_ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    required_context = len(prompt_ids) + len(target_ids)
    if model_context_capacity is not None and required_context > model_context_capacity:
        raise ValueError(f'Full prompt plus training target needs {required_context} tokens, '
                         f'exceeding model context {model_context_capacity}; the rationale was not truncated')
    roles = []
    for token_id, (start, end) in zip(target_ids, offsets):
        if start < label_start + width and end > label_start:
            if (start, end) != (label_start, label_start + width) or token_id != code_ids[label]:
                raise RuntimeError('Label target overlaps another token span')
            roles.append(1)
        elif mode == 'post_cot' and reason and start < width + 1 and end > width:
            if (start, end) != (width, width + 1):
                raise RuntimeError('A token crosses the masked C newline/rationale boundary')
            roles.append(0)
        elif mode == 'pre_cot' and start < marker_end and end > marker_start:
            roles.append(3)
        elif mode == 'think' and end > reason_end:
            roles.append(3)
        elif reason and start < reason_end and end > reason_start:
            roles.append(2)
        else:
            roles.append(0)
    if roles.count(1) != 1 or (not reason and roles.count(2)):
        raise RuntimeError('Incorrect label/rationale target masks')
    if mode == 'think':
        close_id = tokenizer.convert_tokens_to_ids('</think>')
        end_id = tokenizer.convert_tokens_to_ids(THINK_END)
        if target_ids.count(close_id) != 1 or target_ids[-1] != end_id or roles[-1] != 3 or 0 in roles:
            raise RuntimeError('Think target must contain one </think>, end with <|im_end|> and supervise every token')
    positions = [len(prompt_ids) - 1 + index for index, role in enumerate(roles) if role]
    return dict(training_input_ids=list(prompt_ids) + target_ids[:-1],
                target_positions=positions, target_ids=[token for token, role in zip(target_ids, roles) if role],
                target_roles=[role for role in roles if role],
                supervised_counts={'label': roles.count(1), 'rationale': roles.count(2), 'format': roles.count(3)})


def collate_training(tokenizer, items, device):
    import torch
    padded = tokenizer.pad({'input_ids': [item['training_input_ids'] for item in items]}, padding=True, return_tensors='pt')
    width = padded['input_ids'].shape[1]
    positions, targets, roles = [], [], []
    for batch_index, item in enumerate(items):
        offset = width - len(item['training_input_ids'])
        positions.extend((batch_index, position + offset) for position in item['target_positions'])
        targets.extend(item['target_ids'])
        roles.extend(item['target_roles'])
    inputs = {key: value.to(device) for key, value in padded.items()}
    inputs['position_ids'] = (inputs['attention_mask'].cumsum(-1) - 1).clamp_min(0)
    return inputs, torch.tensor(positions, device=device), torch.tensor(targets, device=device), torch.tensor(roles, device=device)


def collate_label_extras(items, device, num_labels=6):
    """Per-target soft label rows [N,K] (zeros = use the hard target) and loss weights [N],
    in the same order as collate_training's targets. Only the label target carries them."""
    import torch
    soft, weights = [], []
    for item in items:
        for role in item['target_roles']:
            if role == 1:
                q = item.get('soft_target')
                soft.append(list(q) if q is not None else [0.0] * num_labels)
                weights.append(float(item.get('loss_weight', 1.0)))
            else:
                soft.append([0.0] * num_labels)
                weights.append(1.0)
    return (torch.tensor(soft, dtype=torch.float32, device=device),
            torch.tensor(weights, dtype=torch.float32, device=device))


def chunked_component_loss(lm_head, hidden, targets, roles, chunk_tokens=32, checkpoint_head=True,
                           soft=None, weights=None, code_ids=None):
    """Exact full-vocabulary CE; checkpoint each head chunk to bound saved logits.

    Optional (label_targets_file): soft [N,6] replaces a target's CE by -sum_c q_c log p(answer_c)
    (full-vocabulary log-softmax read at the six answer tokens) wherever its row sums to > 0;
    weights [N] multiplies every target's loss. Both default to the plain CE above."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    if not hidden.requires_grad:
        raise RuntimeError('Supervised hidden states are detached from the trainable adapters')
    extras = soft is not None
    if extras and (weights is None or code_ids is None or len(soft) != len(targets) or len(weights) != len(targets)):
        raise ValueError('soft targets need aligned weights and the six answer token ids')

    def chunk_loss(states, token_ids, token_roles, *extra):
        logits = lm_head(states).float()
        ce = F.cross_entropy(logits, token_ids, reduction='none')
        if extra:
            token_soft, token_weights = extra
            use = token_soft.sum(-1) > 0
            soft_ce = -(token_soft * logits.log_softmax(-1)[:, code_ids]).sum(-1)
            ce = torch.where(use, soft_ce, ce) * token_weights
        return tuple((ce * (token_roles == role)).sum() for role in (1, 2, 3))

    totals = [hidden.new_zeros((), dtype=torch.float32) for _ in range(3)]
    for start in range(0, len(targets), chunk_tokens):
        args = (hidden[start:start + chunk_tokens], targets[start:start + chunk_tokens], roles[start:start + chunk_tokens])
        if extras:
            args = args + (soft[start:start + chunk_tokens], weights[start:start + chunk_tokens])
        values = checkpoint(chunk_loss, *args, use_reentrant=False) if checkpoint_head else chunk_loss(*args)
        totals = [total + value for total, value in zip(totals, values)]
    return dict(zip(('label', 'rationale', 'format'), totals))


def normalized_group_loss(component_sums, denominators, rationale_weight=1.0, format_weight=1.0):
    """Use whole accumulation-group token counts, including short/empty tails."""
    if denominators['label'] < 1:
        raise ValueError('An accumulation group must contain label targets')
    loss = component_sums['label'] / denominators['label']
    for name, weight in (('rationale', rationale_weight), ('format', format_weight)):
        if denominators[name]:
            loss = loss + weight * component_sums[name] / denominators[name]
    return loss


def capture_adapter_state(model):
    from peft import get_peft_model_state_dict
    return {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(model).items()}


def restore_adapter_state(model, state):
    from peft import set_peft_model_state_dict
    status = set_peft_model_state_dict(model, state, adapter_name='default')
    if status.unexpected_keys or any('lora_' in key for key in status.missing_keys):
        raise RuntimeError('Adapter reload has missing/unexpected LoRA parameters')


def capture_rng_state(include_cuda=True):
    import numpy as np
    import torch
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=dict(algorithm=state[0], keys=state[1].tolist(),
                position=state[2], has_gauss=state[3], cached_gaussian=state[4]),
                torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all() if include_cuda else [])


def restore_rng_state(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    item = state['numpy']
    np.random.set_state((item['algorithm'], np.asarray(item['keys'], dtype=np.uint32),
                         item['position'], item['has_gauss'], item['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def save_training_checkpoint(path, model, optimizer, runtime, best_state, signature, include_cuda=True):
    """Atomically save the same payload used by production and CPU resume audits."""
    import torch
    if any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad):
        raise RuntimeError('Refusing to checkpoint partial gradient accumulation')
    if include_cuda:
        torch.cuda.synchronize()

    def cpu_copy(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: cpu_copy(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu_copy(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cpu_copy(item) for item in value)
        return value

    payload = dict(signature=signature, adapter=capture_adapter_state(model), best_adapter=best_state,
                   optimizer=cpu_copy(optimizer.state_dict()), runtime=runtime, rng=capture_rng_state(include_cuda))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_retry(temporary, path)


def load_training_checkpoint(path, signature, model, optimizer):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('signature') != signature:
        raise RuntimeError('Resume checkpoint signature differs from the current run')
    restore_adapter_state(model, checkpoint['adapter'])
    if checkpoint['runtime']['phase'] == 'training':
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['runtime'], checkpoint['best_adapter'], checkpoint['rng']


def summarize_predictions(predictions, label_names=LABELS):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    if not predictions:
        return dict(samples=0, accuracy=None, macro_f1=None, per_class_f1=None,
                    valid_first_token_rate=None, mean_label_probability_mass=None)
    gold = [row['gold'] for row in predictions]
    pred = [row['predicted'] for row in predictions]
    per_class = f1_score(gold, pred, labels=list(range(len(label_names))), average=None, zero_division=0)
    return dict(samples=len(predictions), accuracy=float(accuracy_score(gold, pred)),
                macro_f1=float(np.mean(per_class)), per_class_f1=dict(zip(label_names, per_class.tolist())),
                valid_first_token_rate=sum(row['legal_first_token'] for row in predictions) / len(predictions),
                format_valid_rate=sum(row.get('format_valid', True) for row in predictions) / len(predictions),
                mean_label_probability_mass=(sum(row['label_probability_mass'] for row in predictions) / len(predictions)
                 if all(row.get('label_probability_mass') is not None for row in predictions) else None))


def run(root, config=None, evaluator=None):
    """Return completed result; compatible interrupted runs resume automatically.

    For B (pre_cot), evaluator is a keyword-only callback accepting model,
    tokenizer, prompts (row_index/input_ids only), label_token_ids,
    generation_config, stage and output_dir. It returns predictions in input
    order with predicted, format_valid, legal_first_token, generated_raw,
    generated_token_count, label_probability_mass and probabilities. Missing
    markers must yield predicted=-1. Gold labels are added only after it returns.
    """
    root = Path(root).expanduser().resolve()
    cfg = dict(DEFAULTS)
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f'Unknown config keys: {sorted(unknown)}')
        cfg.update(config)
    if cfg['mode'] in ('pre_cot', 'think') and (not config or 'run_name' not in config):
        cfg['run_name'] = 'qwen35_pre_cot' if cfg['mode'] == 'pre_cot' else 'qwen35_think'
    if not isinstance(cfg['initial_dev_eval'], bool) or not isinstance(cfg['use_rationales'], bool):
        raise ValueError('initial_dev_eval and use_rationales must be true or false')
    answer_texts = label_texts(cfg['label_format'])
    instruction = instruction_for(cfg['mode'], cfg['label_format'])
    if cfg['instruction_override'] is not None:
        if not isinstance(cfg['instruction_override'], str) or not cfg['instruction_override'].strip():
            raise ValueError('instruction_override must be None or a non-empty string')
        instruction = cfg['instruction_override']
    label_names = LABELS
    if cfg['label_names'] is not None:
        names = cfg['label_names']
        if (not isinstance(names, list) or len(names) < 2 or len(set(names)) != len(names)
                or not all(isinstance(name, str) and name.strip() for name in names)):
            raise ValueError('label_names must be None or a list of at least two distinct non-empty strings')
        if cfg['data_dir'] is None or not isinstance(cfg['answer_words'], list) or len(cfg['answer_words']) != len(names):
            raise ValueError('label_names needs data_dir and answer_words of the same length')
        label_names = list(names)
    num_labels = len(label_names)
    if cfg['answer_words'] is not None:
        words = cfg['answer_words']
        if (not isinstance(words, list) or len(words) != num_labels or len(set(words)) != num_labels
                or not all(isinstance(word, str) and word.strip() and word == word.strip() for word in words)):
            raise ValueError(f'answer_words must be {num_labels} distinct non-empty strings in label order')
        if cfg['mode'] != 'post_cot' or cfg['label_format'] != 'word' or cfg['instruction_override'] is None:
            raise ValueError('answer_words needs post_cot, label_format word and an instruction_override naming the words')
        answer_texts = list(words)
    for key in ('use_dora', 'use_rslora'):
        if not isinstance(cfg[key], bool):
            raise ValueError(f'{key} must be true or false')
    for key in ('loraplus_ratio', 'neftune_alpha', 'rdrop_alpha'):
        if cfg[key] is not None and (isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)) or cfg[key] <= 0):
            raise ValueError(f'{key} must be None or a positive number')
    if (cfg['label_targets_file'] is not None or cfg['demo_file'] is not None or cfg['rdrop_alpha'] is not None) \
            and cfg['mode'] != 'post_cot':
        raise ValueError('label_targets_file, demo_file and rdrop_alpha are implemented for post_cot only')
    if not cfg['use_rationales'] and cfg['mode'] != 'post_cot':
        raise ValueError('Label-only training (use_rationales=false) uses mode post_cot without rationales')
    for key in ('epochs', 'patience', 'batch_size', 'gradient_accumulation', 'eval_batch_size',
                'rank', 'max_text_tokens', 'max_prompt_tokens', 'checkpoint_steps', 'progress_batches',
                'head_chunk_tokens', 'thinking_budget'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('max_rationale_tokens', 'generation_max_new_tokens'):
        if cfg[key] is not None and (type(cfg[key]) is not int or cfg[key] < 1):
            raise ValueError(f'{key} must be None or a positive integer')
    if not isinstance(cfg['run_name'], str) or Path(cfg['run_name']).name != cfg['run_name'] or cfg['run_name'] in ('.', '..'):
        raise ValueError('run_name must be one folder name')
    if cfg['mode'] not in MODES:
        raise ValueError(f'mode must be one of {MODES}')
    if cfg['mode'] in GENERATIVE_MODES and evaluator is None:
        raise ValueError(f'{cfg["mode"]} requires the no-GT generation evaluator callback')
    if cfg['mode'] == 'post_cot' and evaluator is not None:
        raise ValueError('post_cot must classify without a generator callback')
    if min(cfg['rationale_weight'], cfg['format_weight']) < 0:
        raise ValueError('Loss weights must be nonnegative')
    callback_source = inspect.getsourcefile(evaluator) if evaluator is not None else None
    if evaluator is not None and not callback_source:
        raise ValueError('Evaluator must come from an inspectable .py source file')
    if cfg['epochs'] > 3:
        raise ValueError('This experiment permits at most three epochs')
    model_dir = root / 'models' / 'Qwen3.5-4B'
    if cfg['data_dir'] is None:
        data_dir = root / 'dataset_from_friend'
        split_sizes = dict(train=27766, dev=2000, test=5000)
    else:
        if not isinstance(cfg['data_dir'], str):
            raise ValueError('data_dir must be None or a folder path')
        if cfg['use_rationales'] or cfg['demo_file'] is not None:
            raise ValueError('data_dir is implemented for label-only training without demonstrations')
        data_dir = (root / cfg['data_dir']).resolve()
        if not data_dir.is_relative_to(root):
            raise ValueError('data_dir must be within the experiment root')
        split_sizes = json.loads((data_dir / 'split_sizes.json').read_text(encoding='utf-8'))
        if sorted(split_sizes) != ['dev', 'test', 'train'] or not all(type(n) is int and n > 0 for n in split_sizes.values()):
            raise ValueError('split_sizes.json must give positive train/dev/test row counts')
    if cfg['text_prefix'] is not None and not isinstance(cfg['text_prefix'], str):
        raise ValueError('text_prefix must be None or a string')
    text_prefix = '微博文本：\n' if cfg['text_prefix'] is None else cfg['text_prefix']
    declared_context_capacity = context_capacity(json.loads((model_dir / 'config.json').read_text(encoding='utf-8')))
    manifest = json.loads((root / 'model_manifest.json').read_text(encoding='utf-8-sig'))['models']['Qwen3.5-4B']
    package_versions = {name: importlib.metadata.version(name) for name in
                        ('torch', 'transformers', 'peft', 'accelerate', 'numpy', 'scikit-learn')}
    print('Checking input/model file hashes for cache and resume integrity...', flush=True)
    provenance = dict(config=cfg, source_sha256=sha(__file__), model_repo=manifest['repo_id'],
                      model_revision=manifest['revision'],
                      model_sha256={name: sha(model_dir / name) for name in manifest['files']},
                      data_sha256={part: sha(data_dir / f'{part}.tsv') for part in ('train', 'dev', 'test')},
                      packages=package_versions, python=platform.python_version(),
                      label_order=label_names, lora_targets=TARGETS, context_capacity=declared_context_capacity,
                      protocol='fixed original splits; preserve duplicates; select by dev macro-F1; test selected adapter only')
    cleaning = cleaning_path = None
    if cfg['train_cleaning'] is not None:
        if not isinstance(cfg['train_cleaning'], str):
            raise ValueError('train_cleaning must be None or a path to a cleaning spec')
        cleaning_path = (root / cfg['train_cleaning']).resolve()
        if not cleaning_path.is_relative_to(root):
            raise ValueError('Cleaning spec must be within the experiment root')
        cleaning = json.loads(cleaning_path.read_text(encoding='utf-8'))
        if cleaning.get('source_train_sha256') != provenance['data_sha256']['train']:
            raise RuntimeError('Cleaning spec was built from different training data')
    extra_files = {}
    for key in ('label_targets_file', 'demo_file'):
        if cfg[key] is not None:
            if not isinstance(cfg[key], str):
                raise ValueError(f'{key} must be None or a path')
            path = (root / cfg[key]).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f'{key} must be within the experiment root')
            payload = json.loads(path.read_text(encoding='utf-8'))
            if payload.get('source_train_sha256') != provenance['data_sha256']['train']:
                raise RuntimeError(f'{key} was built from different training data')
            extra_files[key] = (path, payload)
    rationale_path = rationale_metadata_path = rationale_metadata = rationale_hash = None
    teacher_kind = None
    if cfg['use_rationales']:
        rationale_path = (root / cfg['rationale_file']).resolve()
        if not rationale_path.is_relative_to(root):
            raise ValueError('Rationale path must be within the experiment root')
        rationale_metadata_path = rationale_path.with_name(rationale_path.stem + '.metadata.json')
        rationale_metadata = json.loads(rationale_metadata_path.read_text(encoding='utf-8'))
        if not rationale_metadata.get('full_train_complete') or rationale_metadata.get('completed_rows') != 27766:
            raise RuntimeError('Rationale generation must be complete for all 27766 training rows')
        rationale_hash = sha(rationale_path)
        if rationale_metadata.get('output_sha256') != rationale_hash:
            raise RuntimeError('Rationale JSONL hash differs from completed metadata')
        teacher = rationale_metadata['provenance']
        if teacher['train_sha256'] != provenance['data_sha256']['train']:
            raise RuntimeError('Teacher training data differ from this experiment')
        teacher_kind = teacher.get('teacher_kind', 'local_qwen')
        if teacher_kind == 'external_api':
            # An API teacher has no local weights; its model, prompt and outputs are pinned instead.
            if not all(teacher.get(key) for key in ('teacher_model', 'prompt_sha256', 'teacher_outputs_sha256')):
                raise RuntimeError('External teacher provenance must name the model, prompt hash and output hash')
        elif teacher['model_files_sha256'] != provenance['model_sha256']:
            raise RuntimeError('Teacher base weights differ from this experiment')
        if teacher['config']['max_text_tokens'] != cfg['max_text_tokens']:
            raise RuntimeError('Teacher/student visible-text token budgets differ')
        # Relabelled rows need rationales written for their new labels: the rationale file
        # must declare this exact cleaning spec (rows are still checked label by label later).
        if cleaning is not None and cleaning['relabel'] and rationale_metadata.get('train_cleaning_sha256') != sha(cleaning_path):
            raise ValueError('Relabelled rows need rationales written for their new labels; '
                             'this rationale file was not built for this cleaning spec')
    from generation_guard import GUARD_POLICY
    provenance.update(rationale_sha256=rationale_hash,
                      rationale_metadata_sha256=sha(rationale_metadata_path) if rationale_metadata_path else None,
                      train_cleaning_sha256=sha(cleaning_path) if cleaning_path else None,
                      label_targets_sha256=sha(extra_files['label_targets_file'][0]) if 'label_targets_file' in extra_files else None,
                      demo_file_sha256=sha(extra_files['demo_file'][0]) if 'demo_file' in extra_files else None,
                      generation_guard_sha256=sha(root / 'generation_guard.py'), generation_guard_policy=GUARD_POLICY,
                      rationale_signature=rationale_metadata['signature'] if rationale_metadata else None,
                      label_format=cfg['label_format'], answer_texts=answer_texts,
                      callback_source_sha256=sha(callback_source) if callback_source else None,
                      callback_identity=f'{evaluator.__module__}.{evaluator.__qualname__}' if evaluator else None,
                      output_order=cfg['mode'], rationale_loss='mean of actual rationale tokens per accumulation group',
                      format_loss='B-only marker tokens, separately normalized; C separator masked',
                      rationale_length_policy='retain full accepted rationale; error on context overflow or optional explicit cap',
                      generation_stop_policy='B: label marker or natural EOS; context ceiling; abnormal repetition/time/memory counted as failed predictions',
                      enable_thinking=cfg['mode'] == 'think', teacher_kind=teacher_kind,
                      thinking_policy=dict(budget_tokens=cfg['thinking_budget'], decoding='greedy',
                                           budget_closure='force </think> at the budget; counted separately',
                                           final_answer='restricted six-way argmax over the answer tokens after </think> and the native separator',
                                           invalid='EOS before </think> or a runtime guard failure scores -1')
                      if cfg['mode'] == 'think' else None,
                      instruction=instruction)
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode('utf-8')).hexdigest()
    out = root / 'runs' / cfg['run_name']
    result_path = out / 'results.json'
    if result_path.exists():
        cached = json.loads(result_path.read_text(encoding='utf-8'))
        if cached.get('signature') != signature:
            raise RuntimeError(f'Completed run inputs/config/code differ: {out}. Use a new run_name.')
        for relative, digest in cached['artifacts_sha256'].items():
            if not (out / relative).is_file() or sha(out / relative) != digest:
                raise RuntimeError(f'Cached artifact missing/modified: {relative}')
        print(f'Using verified completed run: {out}', flush=True)
        return cached
    if out.exists() and any(out.iterdir()):
        previous = out / 'provenance.json'
        if not previous.is_file() or json.loads(previous.read_text(encoding='utf-8')).get('signature') != signature:
            raise RuntimeError(f'Existing run inputs/config/code differ: {out}. Use a new run_name.')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'provenance.json', dict(signature=signature, **provenance))

    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    from peft import LoraConfig, TaskType, get_peft_model
    from sklearn.metrics import accuracy_score, f1_score

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A BF16-capable CUDA GPU is required')
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats()
    session_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise RuntimeError('Tokenizer needs a defined pad token')
    label_ids = [tokenizer.encode(text, add_special_tokens=False) for text in answer_texts]
    if not all(len(ids) == 1 for ids in label_ids) or len({ids[0] for ids in label_ids}) != num_labels:
        raise RuntimeError('Each answer must map to a distinct single token')
    code_id_list = [ids[0] for ids in label_ids]
    code_ids = torch.tensor(code_id_list, device='cuda')
    truncations, prompt_lengths = {}, {}
    rationale_records = load_rationales(rationale_path, rationale_metadata['signature']) if cfg['use_rationales'] else None
    rationale_audit = {'accepted_by_generator': 0, 'used_rationales': 0, 'rejected_or_empty': 0,
                       'max_rationale_tokens': cfg['max_rationale_tokens'], 'rationales_truncated': 0}
    label_targets = demos = None
    if 'label_targets_file' in extra_files:
        label_targets = {}
        for key, rec in extra_files['label_targets_file'][1]['rows'].items():
            q, w = rec.get('q'), rec.get('w', 1.0)
            if q is not None and (len(q) != num_labels or min(q) < 0 or abs(sum(q) - 1) > 1e-4):
                raise ValueError(f'label_targets row {key}: q must be a {num_labels}-way distribution')
            if not isinstance(w, (int, float)) or w <= 0:
                raise ValueError(f'label_targets row {key}: weight must be positive')
            label_targets[int(key)] = dict(q=q, w=float(w))
    if 'demo_file' in extra_files:
        demos = extra_files['demo_file'][1]
        if not all(part in demos for part in ('train', 'dev', 'test')):
            raise ValueError('demo_file needs train/dev/test sections')

    def prepare(rows, part):
        prepared, truncated, lengths = [], 0, []
        for index, row in enumerate(rows):
            text_ids = tokenizer.encode(row['text'], add_special_tokens=False)
            cut = len(text_ids) > cfg['max_text_tokens']
            text = tokenizer.decode(text_ids[:cfg['max_text_tokens']], skip_special_tokens=True) if cut else row['text']
            thinking = cfg['mode'] == 'think'
            demo_block = ''
            if demos is not None:
                shots = demos[part].get(str(row['row_index']), [])
                if not all(isinstance(t, str) and l in range(6) for t, l in shots):
                    raise ValueError(f'{part} row {row["row_index"]}: malformed demonstrations')
                if shots:
                    demo_block = '参考示例（来自训练集）：\n' + '\n'.join(f'微博：{t}\n情绪：{answer_texts[l]}' for t, l in shots) + '\n\n'
            rendered = tokenizer.apply_chat_template(
                [{'role': 'system', 'content': instruction}, {'role': 'user', 'content': demo_block + text_prefix + text}],
                tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
            if thinking and not rendered.endswith('<think>\n'):
                raise RuntimeError('Think prompts must end with the native <think> opener')
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            if not ids or len(ids) > cfg['max_prompt_tokens']:
                raise ValueError(f'{part} row {row["row_index"]}: prompt has {len(ids)} tokens')
            # Check the answer boundary on the first prompts only, instead of
            # encoding K extensions for every dataset row. In think mode the
            # answer follows '</think>\n\n', which the train check below covers.
            if index < 16 and not thinking:
                for label in range(num_labels):
                    if tokenizer.encode(rendered + answer_texts[label], add_special_tokens=False) != ids + label_ids[label]:
                        raise RuntimeError('Answer tokenization differs at the prompt boundary')
            item = dict(row, tokens=ids)
            if part == 'train' and label_targets is not None:
                rec = label_targets.get(row['row_index'])
                if rec is not None:
                    item['soft_target'], item['loss_weight'] = rec['q'], rec['w']
            if part == 'train':
                if rationale_records is None:
                    reason = ''
                else:
                    reason_row = rationale_records[row['row_index']]
                    if reason_row['source_text'] != row['text'] or reason_row['label'] != row['label'] or reason_row['visible_text'] != text:
                        raise RuntimeError(f'Rationale source/visible-text mismatch at row {row["row_index"]}')
                    accepted = bool(reason_row['passes_heuristic_checks'])
                    rationale_audit['accepted_by_generator'] += int(accepted)
                    reason = reason_row['rationale_for_training'] if accepted else ''
                if not isinstance(reason, str):
                    raise RuntimeError('Rationale training target must be a string')
                if cfg['max_rationale_tokens'] is not None and len(tokenizer.encode(reason, add_special_tokens=False)) > cfg['max_rationale_tokens']:
                    raise ValueError(f'Train row {row["row_index"]}: accepted rationale exceeds the explicitly configured '
                                     f'max_rationale_tokens={cfg["max_rationale_tokens"]}; no rationale was dropped or truncated')
                rationale_audit['used_rationales'] += int(bool(reason))
                rationale_audit['rejected_or_empty'] += int(not reason)
                try:
                    item.update(training_sequence(tokenizer, ids, row['label'], reason, cfg['mode'], code_id_list,
                                                  model_context_capacity=declared_context_capacity,
                                                  label_text=answer_texts[row['label']]))
                except ValueError as exc:
                    raise ValueError(f'Train row {row["row_index"]}: {exc}') from exc
                if thinking:
                    # The supervised sequence must equal the canonical tokenization of
                    # prompt + rationale + '\n</think>\n\n' + answer + '<|im_end|>'.
                    full = tokenizer.encode(rendered + reason + THINK_CLOSE + answer_texts[row['label']] + THINK_END,
                                            add_special_tokens=False)
                    if full != item['training_input_ids'] + [tokenizer.convert_tokens_to_ids(THINK_END)]:
                        raise RuntimeError(f'Train row {row["row_index"]}: think target differs from canonical tokenization')
            prepared.append(item)
            truncated += int(cut)
            lengths.append(len(ids))
        truncations[part] = truncated
        prompt_lengths[part] = dict(min=min(lengths), max=max(lengths), mean=sum(lengths) / len(lengths))
        return prepared

    # Test bytes were hashed above, but test texts and labels are not parsed yet.
    train_rows = read_rows(data_dir / 'train.tsv', split_sizes['train'], num_labels)
    if cleaning is not None:
        train_rows = apply_cleaning(train_rows, cleaning, num_labels)
    dev_rows = read_rows(data_dir / 'dev.tsv', split_sizes['dev'], num_labels)
    duplicates = dict(train_dev=duplicate_summary(train_rows, dev_rows),
                      train_extra_duplicate_rows=len(train_rows) - len({r['text'] for r in train_rows}),
                      dev_extra_duplicate_rows=len(dev_rows) - len({r['text'] for r in dev_rows}),
                      action='preserved; no row removed from any split' if cleaning is None else
                      f'train cleaned in memory by {cfg["train_cleaning"]} (drop {len(cleaning["drop"])}, '
                      f'relabel {len(cleaning["relabel"])}); dev/test untouched; duplicates otherwise preserved')
    write_json(out / 'duplicate_audit.json', duplicates)
    train_data = prepare(train_rows, 'train')
    dev_data = prepare(dev_rows, 'dev')
    write_json(out / 'rationale_audit.json', rationale_audit)
    print(f'{cfg["mode"]}: {len(train_data)} train, {len(dev_data)} dev; microbatch={cfg["batch_size"]}, effective batch={cfg["batch_size"] * cfg["gradient_accumulation"]}', flush=True)
    model = optimizer = trainable = None
    try:
        model, loading = Qwen3_5ForCausalLM.from_pretrained(
            model_dir, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa',
            device_map={'': 'cuda'}, output_loading_info=True)
        if context_capacity(model.config) != declared_context_capacity:
            raise RuntimeError('Loaded model context capacity differs from the signed local config')
        loading = json_friendly(loading)
        bad_extra = [key for key in loading.get('unexpected_keys', [])
                     if not any(fragment in key for fragment in ('visual.', 'vision.', 'mtp.'))]
        if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs') or bad_extra:
            raise RuntimeError(f'Language backbone load audit failed: {loading}')
        model.config.use_cache = False
        lora_kwargs = dict(task_type=TaskType.CAUSAL_LM, r=cfg['rank'], lora_alpha=cfg['lora_alpha'],
                           lora_dropout=cfg['dropout'], target_modules=TARGETS)
        if cfg['use_dora']:
            lora_kwargs['use_dora'] = True      # DoRA (Liu et al., ICML 2024): magnitude + direction
        if cfg['use_rslora']:
            lora_kwargs['use_rslora'] = True    # rank-stabilized scaling alpha / sqrt(r)
        model = get_peft_model(model, LoraConfig(**lora_kwargs))
        if cfg['neftune_alpha'] is not None:
            # NEFTune (Jain et al., ICLR 2024): uniform noise on input embeddings, training only.
            def neftune_hook(module, inputs, output, alpha=float(cfg['neftune_alpha'])):
                if module.training:
                    scale = alpha / math.sqrt(output.size(1) * output.size(2))
                    output = output + torch.zeros_like(output).uniform_(-scale, scale)
                return output
            model.get_base_model().get_input_embeddings().register_forward_hook(neftune_hook)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
        if not trainable or not all('lora_' in name for name in trainable):
            raise RuntimeError('Expected only trainable LoRA parameters')
        covered = sorted({name.split('.lora_A.')[0] for name in trainable if '.lora_A.' in name})
        if set(TARGETS) - {name.rsplit('.', 1)[-1] for name in covered}:
            raise RuntimeError('Some requested LoRA projection types are missing')
        watched = {kind: next((name for name in trainable if 'lora_B' in name and kind in name), None)
                   for kind in ('linear_attn', 'self_attn')}
        if not all(watched.values()):
            raise RuntimeError('LoRA must cover linear attention and full attention')
        if cfg['loraplus_ratio'] is not None:
            # LoRA+ (Hayou et al., ICML 2024): B matrices learn ratio x faster than A matrices.
            groups = [dict(params=[p for n, p in trainable.items() if 'lora_B' not in n], lr=cfg['learning_rate']),
                      dict(params=[p for n, p in trainable.items() if 'lora_B' in n], lr=cfg['learning_rate'] * cfg['loraplus_ratio'])]
            optimizer = torch.optim.AdamW(groups, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        else:
            optimizer = torch.optim.AdamW(list(trainable.values()), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        if {id(p) for group in optimizer.param_groups for p in group['params']} != {id(p) for p in trainable.values()}:
            raise RuntimeError('Optimizer coverage mismatch')
        environment = dict(packages=package_versions, python=platform.python_version(), platform=platform.platform(),
                           gpu=torch.cuda.get_device_name(0), cuda_runtime=torch.version.cuda,
                           precision='BF16 base; PEFT adapter dtype recorded below',
                           adapter_dtypes=sorted({str(p.dtype) for p in trainable.values()}))
        write_json(out / 'environment.json', environment)
        write_json(out / 'loading_audit.json', loading)
        write_json(out / 'lora_modules.json', covered)

        causal_model = model.get_base_model()
        if not hasattr(causal_model, 'model') or not hasattr(causal_model, 'lm_head'):
            raise RuntimeError('Unexpected Qwen causal backbone layout')

        def memory():
            return dict(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)

        def logits_for(items):
            batch = tokenizer.pad({'input_ids': [item['tokens'] for item in items]}, padding=True, return_tensors='pt')
            ids = batch['input_ids'].to('cuda')
            mask = batch['attention_mask'].to('cuda')
            position_ids = (mask.cumsum(-1) - 1).clamp_min(0)
            return model(input_ids=ids, attention_mask=mask, position_ids=position_ids,
                         use_cache=False, logits_to_keep=1).logits[:, -1, :].float()

        @torch.inference_mode()
        def evaluate(items, description):
            model.eval()
            torch.cuda.synchronize()
            tick = time.perf_counter()
            if cfg['mode'] in GENERATIVE_MODES:
                no_gold_prompts = [{'row_index': item['row_index'], 'input_ids': list(item['tokens'])} for item in items]
                generation_config = ({'batch_size': cfg['eval_batch_size'], 'max_new_tokens': cfg['thinking_budget'], 'enable_thinking': True}
                                     if cfg['mode'] == 'think' else
                                     {'batch_size': cfg['eval_batch_size'], 'max_new_tokens': cfg['generation_max_new_tokens'], 'enable_thinking': False})
                response = evaluator(model=model, tokenizer=tokenizer, prompts=no_gold_prompts,
                                     label_token_ids=list(code_id_list), generation_config=generation_config,
                                     stage=description, output_dir=out)
                raw_predictions = response['predictions']
                if [row['row_index'] for row in raw_predictions] != [item['row_index'] for item in items]:
                    raise RuntimeError('Generation callback returned wrong/missing row indices')
                predictions = []
                for item, raw in zip(items, raw_predictions):
                    if raw.get('predicted') not in (-1, 0, 1, 2, 3, 4, 5) or (not raw.get('format_valid') and raw['predicted'] != -1):
                        raise RuntimeError('Invalid generation must be scored as prediction -1')
                    if {'gold', 'label', 'true_label'} & set(raw):
                        raise RuntimeError('Generation callback must not return gold labels')
                    predictions.append(dict(raw, text=item['text'], gold=item['label'], gold_label=LABELS[item['label']],
                                            predicted_label=LABELS[raw['predicted']] if raw['predicted'] >= 0 else 'INVALID'))
                torch.cuda.synchronize()
                return dict(summarize_predictions(predictions), full_vocabulary_ce=None,
                            generated_token_count=sum(row['generated_token_count'] for row in predictions),
                            generation_summary=response.get('summary'),
                            elapsed_seconds=time.perf_counter() - tick, **memory()), predictions
            predictions, loss_sum = [], 0.0
            for start in range(0, len(items), cfg['eval_batch_size']):
                selected = items[start:start + cfg['eval_batch_size']]
                logits = logits_for(selected)
                if not torch.isfinite(logits).all():
                    raise RuntimeError(f'Non-finite logits during {description}')
                legal_logits = logits[:, code_ids]
                probs = legal_logits.softmax(-1).cpu().tolist()
                classes = legal_logits.argmax(-1).cpu().tolist()
                raw_tokens = logits.argmax(-1).cpu().tolist()
                masses = (torch.logsumexp(legal_logits, -1) - torch.logsumexp(logits, -1)).exp().cpu().tolist()
                targets = code_ids[torch.tensor([item['label'] for item in selected], device='cuda')]
                loss_sum += float(F.cross_entropy(logits, targets, reduction='sum'))
                for item, pred, probability, raw, mass in zip(selected, classes, probs, raw_tokens, masses):
                    predictions.append(dict(row_index=item['row_index'], text=item['text'], gold=item['label'],
                                            gold_label=label_names[item['label']], predicted=pred, predicted_label=label_names[pred],
                                            probabilities=probability, raw_argmax_token_id=raw,
                                            legal_first_token=raw in code_id_list, label_probability_mass=mass))
                batch_number = start // cfg['eval_batch_size'] + 1
                if batch_number % cfg['progress_batches'] == 0 or len(predictions) == len(items):
                    torch.cuda.synchronize()
                    progress = dict(phase=description, rows=len(predictions), total_rows=len(items),
                                    elapsed_seconds=time.perf_counter() - tick, **memory())
                    write_json(out / 'evaluation_progress.json', progress)
                    print(f'{description}: {len(predictions)}/{len(items)} rows, {progress["elapsed_seconds"]:.1f}s', flush=True)
            torch.cuda.synchronize()
            metrics = dict(summarize_predictions(predictions, label_names), full_vocabulary_ce=loss_sum / len(items),
                           elapsed_seconds=time.perf_counter() - tick, **memory())
            return metrics, predictions

        runtime = dict(phase='training', epoch=1, next_batch=0, optimizer_steps=0,
                       epoch_loss_sum=0.0, epoch_samples=0, epoch_train_seconds=0.0,
                       total_train_seconds=0.0, best_epoch=None, best_dev=None, best_predictions=None,
                       bad_epochs=0, history=[], initial_dev=None, gradient_update_audit=None,
                       recent_batch_losses=[], recent_batch_seconds=[], resume_count=0,
                       component_loss_sums={'label': 0.0, 'rationale': 0.0, 'format': 0.0},
                       component_token_counts={'label': 0, 'rationale': 0, 'format': 0},
                       total_supervised_tokens={'label': 0, 'rationale': 0, 'format': 0},
                       group_denominator_history=[])
        best_state = None
        resume_path = out / 'resume.pt'
        resumed_rng = None
        if resume_path.exists():
            # This is our own atomic checkpoint; its contents use only tensors
            # and built-in containers so safe weights-only loading is sufficient.
            runtime, best_state, resumed_rng = load_training_checkpoint(resume_path, signature, model, optimizer)
            runtime['resume_count'] += 1
            print(f'Resuming {runtime["phase"]}, epoch={runtime["epoch"]}, next_batch={runtime["next_batch"]}, optimizer_steps={runtime["optimizer_steps"]}', flush=True)

        def save_checkpoint():
            save_training_checkpoint(resume_path, model, optimizer, runtime, best_state, signature)
            write_json(out / 'resume_status.json', dict(signature=signature, phase=runtime['phase'],
                       epoch=runtime['epoch'], next_batch=runtime['next_batch'], optimizer_steps=runtime['optimizer_steps'],
                       best_epoch=runtime['best_epoch'], checkpoint_bytes=resume_path.stat().st_size))

        optimizer.zero_grad(set_to_none=True)
        if runtime['initial_dev'] is None:
            if cfg['initial_dev_eval']:
                metrics, predictions = evaluate(dev_data, 'initial dev')
                runtime['initial_dev'] = metrics
                write_json(out / 'initial_dev_predictions.json', predictions)
                print(f'Initial dev Acc={metrics["accuracy"]:.4f}, MF1={metrics["macro_f1"]:.4f}', flush=True)
            else:
                runtime['initial_dev'] = dict(skipped=True, reason='initial_dev_eval=false; untrained baselines are reported by separate runs')
                print('Initial dev evaluation skipped by configuration', flush=True)
            save_checkpoint()
        if resumed_rng is not None:
            restore_rng_state(resumed_rng)
            del resumed_rng
        watched_before = {name: trainable[name].detach().clone() for name in watched.values()} if runtime['gradient_update_audit'] is None else None

        while runtime['phase'] == 'training':
            epoch = runtime['epoch']
            batches = epoch_batches(len(train_data), cfg['batch_size'], cfg['seed'], epoch)
            start_batch = runtime['next_batch']
            if not 0 <= start_batch <= len(batches) or (start_batch != len(batches) and start_batch % cfg['gradient_accumulation']):
                raise RuntimeError('Resume cursor is not a valid accumulation boundary')
            model.train()
            for batch_index in range(start_batch, len(batches)):
                torch.cuda.synchronize()
                tick = time.perf_counter()
                items = [train_data[index] for index in batches[batch_index]]
                group_start = batch_index // cfg['gradient_accumulation'] * cfg['gradient_accumulation']
                group_end = min(group_start + cfg['gradient_accumulation'], len(batches))
                group_items = [train_data[index] for j in range(group_start, group_end) for index in batches[j]]
                denominators = {name: sum(item['supervised_counts'][name] for item in group_items)
                                for name in ('label', 'rationale', 'format')}
                input_batch, positions, target_ids, roles = collate_training(tokenizer, items, device='cuda')
                hidden = causal_model.model(**input_batch, use_cache=False).last_hidden_state
                supervised_hidden = hidden[positions[:, 0], positions[:, 1], :]
                if label_targets is not None:
                    soft_rows, loss_weights = collate_label_extras(items, 'cuda', num_labels)
                    component_sums = chunked_component_loss(causal_model.lm_head, supervised_hidden, target_ids, roles,
                                                           cfg['head_chunk_tokens'], checkpoint_head=True,
                                                           soft=soft_rows, weights=loss_weights, code_ids=code_ids)
                else:
                    component_sums = chunked_component_loss(causal_model.lm_head, supervised_hidden, target_ids, roles,
                                                           cfg['head_chunk_tokens'], checkpoint_head=True)
                loss = normalized_group_loss(component_sums, denominators, cfg['rationale_weight'], cfg['format_weight'])
                if cfg['rdrop_alpha'] is not None:
                    # R-Drop (Liang et al., NeurIPS 2021): a second dropout pass; symmetric KL between the
                    # two restricted K-way answer distributions at the label position.
                    second = causal_model.model(**input_batch, use_cache=False).last_hidden_state
                    label_pos = positions[roles == 1]
                    first_lp = causal_model.lm_head(hidden[label_pos[:, 0], label_pos[:, 1], :]).float()[:, code_ids].log_softmax(-1)
                    second_lp = causal_model.lm_head(second[label_pos[:, 0], label_pos[:, 1], :]).float()[:, code_ids].log_softmax(-1)
                    kl = 0.5 * (F.kl_div(first_lp, second_lp, log_target=True, reduction='sum')
                                + F.kl_div(second_lp, first_lp, log_target=True, reduction='sum'))
                    loss = loss + cfg['rdrop_alpha'] * kl / denominators['label']
                    del second, first_lp, second_lp
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at epoch={epoch}, batch={batch_index}')
                loss.backward()
                detached_sums = {name: float(value.detach()) for name, value in component_sums.items()}
                for name in denominators:
                    count = sum(item['supervised_counts'][name] for item in items)
                    runtime['component_loss_sums'][name] += detached_sums[name]
                    runtime['component_token_counts'][name] += count
                    runtime['total_supervised_tokens'][name] += count
                boundary = batch_index + 1 == group_end
                if boundary:
                    torch.nn.utils.clip_grad_norm_(list(trainable.values()), cfg['max_grad_norm'], error_if_nonfinite=True)
                    gradients = {name: float(trainable[name].grad.float().norm()) for name in watched.values()} if watched_before is not None else None
                    optimizer.step()
                    if watched_before is not None:
                        audit = {kind: dict(parameter=name, gradient_norm=gradients[name],
                                            max_abs_update=float((trainable[name].detach() - watched_before[name]).abs().max()))
                                 for kind, name in watched.items()}
                        if any(row['gradient_norm'] <= 0 or row['max_abs_update'] <= 0 for row in audit.values()):
                            raise RuntimeError(f'LoRA update sanity check failed: {audit}')
                        runtime['gradient_update_audit'] = audit
                        write_json(out / 'gradient_update_audit.json', audit)
                        watched_before = None
                    optimizer.zero_grad(set_to_none=True)
                    runtime['optimizer_steps'] += 1
                    runtime['group_denominator_history'].append(dict(epoch=epoch, start_batch=group_start,
                        end_batch=group_end, optimizer_step=runtime['optimizer_steps'], **denominators))
                torch.cuda.synchronize()
                duration = time.perf_counter() - tick
                scalar_loss = detached_sums['label'] / len(items)
                for name, weight in (('rationale', cfg['rationale_weight']), ('format', cfg['format_weight'])):
                    token_count = sum(item['supervised_counts'][name] for item in items)
                    if token_count:
                        scalar_loss += weight * detached_sums[name] / token_count
                del hidden, supervised_hidden, input_batch, positions, target_ids, roles, component_sums
                runtime['next_batch'] = batch_index + 1
                runtime['epoch_loss_sum'] += scalar_loss * len(items)
                runtime['epoch_samples'] += len(items)
                runtime['epoch_train_seconds'] += duration
                runtime['total_train_seconds'] += duration
                runtime['recent_batch_losses'] = (runtime['recent_batch_losses'] + [scalar_loss])[-cfg['progress_batches']:]
                runtime['recent_batch_seconds'] = (runtime['recent_batch_seconds'] + [duration])[-cfg['progress_batches']:]
                if (batch_index + 1) % cfg['progress_batches'] == 0 or batch_index + 1 == len(batches):
                    progress = dict(epoch=epoch, batch=batch_index + 1, total_batches=len(batches),
                                    samples=runtime['epoch_samples'], optimizer_steps=runtime['optimizer_steps'],
                                    mean_train_loss=runtime['epoch_loss_sum'] / runtime['epoch_samples'],
                                    recent_loss=sum(runtime['recent_batch_losses']) / len(runtime['recent_batch_losses']),
                                    recent_seconds_per_batch=sum(runtime['recent_batch_seconds']) / len(runtime['recent_batch_seconds']),
                                    epoch_train_seconds=runtime['epoch_train_seconds'], **memory())
                    write_json(out / 'progress.json', progress)
                    print(f'Epoch {epoch}/{cfg["epochs"]}, batch {batch_index + 1}/{len(batches)}, loss={progress["recent_loss"]:.4f}, sec/batch={progress["recent_seconds_per_batch"]:.3f}, peak={progress["peak_allocated_gib"]:.2f}GiB', flush=True)
                if boundary and (runtime['optimizer_steps'] % cfg['checkpoint_steps'] == 0 or batch_index + 1 == len(batches)):
                    save_checkpoint()
            if runtime['epoch_samples'] != len(train_data):
                raise RuntimeError('Epoch sample accounting mismatch')
            metrics, predictions = evaluate(dev_data, f'epoch {epoch} dev')
            component_means = {name: (runtime['component_loss_sums'][name] / runtime['component_token_counts'][name]
                                      if runtime['component_token_counts'][name] else 0.0)
                               for name in runtime['component_loss_sums']}
            runtime['history'].append(dict(epoch=epoch, train_loss=component_means['label'] + cfg['rationale_weight'] * component_means['rationale'] + cfg['format_weight'] * component_means['format'],
                                           component_losses=component_means, supervised_tokens=dict(runtime['component_token_counts']),
                                           train_seconds=runtime['epoch_train_seconds'], optimizer_steps=runtime['optimizer_steps'], dev=metrics))
            if runtime['best_dev'] is None or metrics['macro_f1'] > runtime['best_dev']['macro_f1']:
                runtime['best_epoch'], runtime['best_dev'], runtime['best_predictions'] = epoch, metrics, predictions
                runtime['bad_epochs'] = 0
                best_state = capture_adapter_state(model)
                model.save_pretrained(out / 'best_adapter', safe_serialization=True, save_embedding_layers=False)
                tokenizer.save_pretrained(out / 'best_adapter')
                write_json(out / 'dev_predictions.json', predictions)
            else:
                runtime['bad_epochs'] += 1
            write_json(out / 'history.json', dict(initial_dev=runtime['initial_dev'], epochs=runtime['history']))
            print(f'Epoch {epoch}: dev Acc={metrics["accuracy"]:.4f}, MF1={metrics["macro_f1"]:.4f}; best epoch={runtime["best_epoch"]}', flush=True)
            if runtime['bad_epochs'] >= cfg['patience'] or epoch >= cfg['epochs']:
                runtime['phase'] = 'train_done'
                runtime['stop_reason'] = 'early_stopping' if runtime['bad_epochs'] >= cfg['patience'] else 'max_epochs'
            else:
                runtime.update(epoch=epoch + 1, next_batch=0, epoch_loss_sum=0.0, epoch_samples=0,
                               epoch_train_seconds=0.0, recent_batch_losses=[], recent_batch_seconds=[],
                               component_loss_sums={'label': 0.0, 'rationale': 0.0, 'format': 0.0},
                               component_token_counts={'label': 0, 'rationale': 0, 'format': 0})
            save_checkpoint()

        if best_state is None or runtime['best_epoch'] is None:
            raise RuntimeError('No trained dev-selected adapter exists')
        restore_adapter_state(model, best_state)
        # Re-export from the checkpointed best state; this also repairs any
        # interrupted multi-file save of the human-readable HF adapter folder.
        model.save_pretrained(out / 'best_adapter', safe_serialization=True, save_embedding_layers=False)
        tokenizer.save_pretrained(out / 'best_adapter')
        restored_dev, dev_predictions = evaluate(dev_data, 'restored best dev')
        reload_agreement = sum(a['predicted'] == b['predicted'] for a, b in zip(dev_predictions, runtime['best_predictions'])) / len(dev_predictions)
        # One forward pass must reproduce exactly. Batched generation can flip a
        # rare borderline token after a reload, so generative modes allow 1%.
        if reload_agreement < (0.99 if cfg['mode'] in GENERATIVE_MODES else 1.0):
            raise RuntimeError(f'Restored best adapter dev predictions differ (agreement {reload_agreement:.4f})')
        write_json(out / 'dev_predictions.json', dev_predictions)
        # Model selection has finished. This is the first access to test text or
        # labels; the test file was only hashed before this point.
        test_rows = read_rows(data_dir / 'test.tsv', split_sizes['test'], num_labels)
        test_data = prepare(test_rows, 'test')
        duplicates.update(train_test=duplicate_summary(train_rows, test_rows), dev_test=duplicate_summary(dev_rows, test_rows),
                          test_extra_duplicate_rows=len(test_rows) - len({row['text'] for row in test_rows}))
        write_json(out / 'duplicate_audit.json', duplicates)
        test_metrics, test_predictions = evaluate(test_data, 'selected adapter test')
        write_json(out / 'test_predictions.json', test_predictions)
        train_texts = {row['text'] for row in train_rows}
        seen = [row for row in test_predictions if row['text'] in train_texts]
        unseen = [row for row in test_predictions if row['text'] not in train_texts]
        write_json(out / 'group_denominators.json', runtime['group_denominator_history'])
        result = dict(signature=signature, provenance=provenance, status='complete', environment=environment,
                      mode=cfg['mode'], rationale_audit=rationale_audit, total_supervised_tokens=runtime['total_supervised_tokens'],
                      inference_generates_rationale=cfg['mode'] in GENERATIVE_MODES,
                      kind='full_data_rationale_supervision' if cfg['use_rationales'] else 'full_data_label_only',
                      label_format=cfg['label_format'], answer_texts=answer_texts,
                      train_size=len(train_data), dev_size=len(dev_data), test_size=len(test_data),
                      label_order=label_names, label_token_ids=code_id_list,
                      evaluation=('C: one-forward restricted label; B: generated reason then restricted label, missing marker scored invalid; '
                                  'think: greedy native thinking (budget-forced </think> counted), then restricted label after the separator; '
                                  'EOS before </think> or guard failure scored invalid; full-vocabulary target CE'),
                      initial_dev=runtime['initial_dev'], history=runtime['history'], best_epoch=runtime['best_epoch'],
                      best_dev=restored_dev, test=test_metrics, test_train_seen=summarize_predictions(seen, label_names),
                      test_train_unseen=summarize_predictions(unseen, label_names),
                      stop_reason=runtime['stop_reason'], selection='dev Macro-F1; strict improvement; earliest epoch wins ties; initial excluded',
                      test_checkpoint_count=1, adapter_reload_dev_prediction_agreement=reload_agreement, duplicates=duplicates,
                      gradient_update_audit=runtime['gradient_update_audit'], optimizer_steps=runtime['optimizer_steps'],
                      resume_count=runtime['resume_count'], trainable_parameters=sum(p.numel() for p in trainable.values()),
                      lora_module_count=len(covered), truncations=truncations, prompt_lengths=prompt_lengths,
                      train_seconds=runtime['total_train_seconds'], current_session_seconds=time.perf_counter() - session_started,
                      checkpoint_policy='atomic resume.pt at optimizer boundaries; exact input/config/source signature; all RNG states restored',
                      **memory())
        result['artifacts_sha256'] = {path.relative_to(out).as_posix(): sha(path) for path in out.rglob('*')
                                      if path.is_file() and path.name != 'results.json' and not path.name.endswith('.tmp')}
        write_json(result_path, result)
        print(f'Completed: best epoch {result["best_epoch"]}, test Acc={test_metrics["accuracy"]:.4f}, MF1={test_metrics["macro_f1"]:.4f}', flush=True)
        return result
    finally:
        # No model or optimizer is returned to the caller; release GPU memory.
        logits = loss = targets = watched_before = None
        hidden = supervised_hidden = input_batch = positions = target_ids = roles = component_sums = None
        causal_model = None
        optimizer = None
        trainable = None
        model = None
        code_ids = None
        gc.collect()
        torch.cuda.empty_cache()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--config', type=Path, help='Optional JSON overrides of DEFAULTS')
    parser.add_argument('--mode', choices=MODES, help='Override the experiment mode')
    args = parser.parse_args(argv)
    overrides = json.loads(args.config.read_text(encoding='utf-8-sig')) if args.config else {}
    if args.mode:
        overrides['mode'] = args.mode
    evaluator = None
    mode = overrides.get('mode', DEFAULTS['mode'])
    if mode == 'pre_cot':
        from reasoning_evaluation import evaluate_explicit
        evaluator = evaluate_explicit
    elif mode == 'think':
        from thinking_sft_evaluation import evaluate_thinking
        evaluator = evaluate_thinking
    return run(args.root, overrides, evaluator=evaluator)


if __name__ == '__main__':
    main()
