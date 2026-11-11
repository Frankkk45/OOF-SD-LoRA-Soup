"""Per-row abnormal-generation protection, without changing token scores.

Normal EOS and an optional final-marker predicate take priority. There is no
ordinary output-token cap. The elapsed-time check is cooperative at token
boundaries; a separate process watchdog is needed for a stalled model forward.
This module does not import Torch/Transformers or initialize CUDA on import.
"""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time


GUARD_POLICY = {
    'version': 1,
    'repeat_min_period_tokens': 1,
    'repeat_max_period_tokens': 64,
    'repeat_min_cycles': 8,
    'repeat_min_total_tokens': 128,
    'batch_timeout_seconds': 120.0,
    'minimum_cuda_available_bytes': 1024 ** 3,
    'heartbeat_interval_seconds': 10.0,
    'normal_termination_priority': 'EOS, then final-marker predicate, before abnormal guards',
    'cuda_available_estimate': 'free + max(0, reserved - allocated); heuristic, not an OOM guarantee',
    'time_limit_scope': 'cooperative token-boundary check; external watchdog handles stalled forwards',
    'failure_policy': 'teacher: empty reason, retain label; B evaluation: prediction -1, retain row',
}


def replace_with_retry(source, target, timeout_seconds=5.0, poll_seconds=0.05):
    """Atomically replace a status file, tolerating short Windows reader locks.

    Only Windows access/share/lock PermissionErrors are retried. The existing
    target is never deleted first, and persistent errors remain visible after
    the monotonic deadline. This changes file I/O only, not generation policy.
    """
    if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
        raise ValueError('timeout_seconds must be finite and nonnegative')
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError('poll_seconds must be finite and positive')
    source, target = Path(source), Path(target)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            return source.replace(target)
        except PermissionError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(poll_seconds, remaining))


def repeated_suffix(token_ids):
    """Return evidence for an exact short-period suffix, or None."""
    minimum = GUARD_POLICY['repeat_min_total_tokens']
    if len(token_ids) < minimum:
        return None
    for period in range(GUARD_POLICY['repeat_min_period_tokens'], GUARD_POLICY['repeat_max_period_tokens'] + 1):
        cycles = max(GUARD_POLICY['repeat_min_cycles'], math.ceil(minimum / period))
        count = period * cycles
        if len(token_ids) >= count:
            block = token_ids[-period:]
            if token_ids[-count:] == block * cycles:
                return dict(period_tokens=period, consecutive_cycles=cycles,
                            repeated_suffix_tokens=count, pattern_token_ids=list(block))
    return None


class RuntimeGenerationGuard:
    """A callable suitable for Transformers StoppingCriteriaList.

    ``stop_predicate`` receives one row's generated token IDs, excluding prompt
    and any EOS. ``failures`` and ``stopped_token_counts`` use batch indices.
    The latter freezes the original generated length on its first termination,
    including EOS if EOS caused the stop. Callers must slice to this length
    before parsing, so later HF padding cannot be mistaken for a normal EOS.
    Call ``finish(state='complete')`` after generate returns, or ``'failed'``
    when it raises. Heartbeat write/flush errors are deliberately propagated.
    """

    def __init__(self, prompt_width, eos_ids, pad_id, row_ids, stop_predicate=None,
                 heartbeat_path=None, stage='generation'):
        if type(prompt_width) is not int or prompt_width < 1:
            raise ValueError('prompt_width must be a positive integer')
        if not row_ids:
            raise ValueError('row_ids must contain at least one row')
        self.prompt_width = prompt_width
        self.eos_ids = {int(token) for token in eos_ids if token is not None}
        if not self.eos_ids:
            raise ValueError('At least one EOS ID is required')
        self.pad_id = pad_id
        self.row_ids = list(row_ids)
        self.stop_predicate = stop_predicate
        self.heartbeat_path = Path(heartbeat_path) if heartbeat_path is not None else None
        self.stage = str(stage)
        self.failures = {}
        self.normal_stops = {}
        self.stopped_token_counts = {}
        self.generated_token_counts = [0] * len(self.row_ids)
        self._started = time.monotonic()
        self._last_heartbeat = None
        self._finished = False
        self._last_memory = None
        self._write_heartbeat(state='active', force=True)

    def _elapsed(self):
        return max(0.0, time.monotonic() - self._started)

    def _write_heartbeat(self, state, force=False):
        now = time.monotonic()
        if not force and self._last_heartbeat is not None and now - self._last_heartbeat < GUARD_POLICY['heartbeat_interval_seconds']:
            return
        if self.heartbeat_path is not None:
            ended = len(self.stopped_token_counts)
            payload = dict(state=state, pid=os.getpid(), updated_utc=datetime.now(timezone.utc).isoformat(),
                stage=self.stage, elapsed_seconds=max(0.0, now - self._started), row_ids=self.row_ids,
                batch_size=len(self.row_ids), active_rows=len(self.row_ids) - ended, ended_rows=ended,
                failed_rows=len(self.failures), generated_token_counts=list(self.generated_token_counts),
                stopped_token_counts=dict(self.stopped_token_counts), failures=dict(self.failures),
                normal_stops=dict(self.normal_stops), cuda_memory=self._last_memory,
                guard_policy=GUARD_POLICY)
            self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.heartbeat_path.with_name(self.heartbeat_path.name + '.tmp')
            with temporary.open('w', encoding='utf-8') as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            replace_with_retry(temporary, self.heartbeat_path)
        self._last_heartbeat = now

    def _normal_stop(self, index, count, reason, stop_token_id=None):
        self.stopped_token_counts[index] = count
        self.generated_token_counts[index] = count
        self.normal_stops[index] = dict(reason=reason, row_id=self.row_ids[index], token_count=count,
                                       elapsed_seconds=self._elapsed(), stop_token_id=stop_token_id)

    def _failure(self, index, count, reason, details=None):
        self.stopped_token_counts[index] = count
        self.generated_token_counts[index] = count
        self.failures[index] = dict(reason=reason, row_id=self.row_ids[index], token_count=count,
                                   elapsed_seconds=self._elapsed(), details=details or {})

    def _memory_snapshot(self, device):
        if device.type != 'cuda':
            return None
        import torch
        free, total = torch.cuda.mem_get_info(device)
        reserved = torch.cuda.memory_reserved(device)
        allocated = torch.cuda.memory_allocated(device)
        return dict(free_bytes=int(free), total_bytes=int(total), reserved_bytes=int(reserved),
                    allocated_bytes=int(allocated),
                    available_estimate_bytes=int(free + max(0, reserved - allocated)))

    def __call__(self, input_ids, scores=None, **kwargs):
        import torch
        if self._finished:
            raise RuntimeError('A finished generation guard cannot be reused for another batch')
        if input_ids.ndim != 2 or input_ids.shape[0] != len(self.row_ids) or input_ids.shape[1] < self.prompt_width:
            raise ValueError('Generation tensor shape does not match this guard batch')
        count = input_ids.shape[1] - self.prompt_width
        # Only the generated suffix is inspected. Prompt padding/special tokens
        # cannot trigger a stop; already-ended rows are never re-inspected.
        suffixes = input_ids[:, self.prompt_width:].detach().cpu().tolist()
        ended_before = len(self.stopped_token_counts)
        for index, suffix in enumerate(suffixes):
            if index in self.stopped_token_counts:
                continue
            if count < self.generated_token_counts[index]:
                raise ValueError('Generated suffix shrank within a batch')
            self.generated_token_counts[index] = count
            eos_position = next((i for i, token in enumerate(suffix) if token in self.eos_ids), None)
            if eos_position is not None:
                self._normal_stop(index, eos_position + 1, 'eos', suffix[eos_position])
                continue
            if self.stop_predicate is not None and self.stop_predicate(suffix):
                self._normal_stop(index, count, 'stop_predicate')
                continue
            repetition = repeated_suffix(suffix)
            if repetition is not None:
                self._failure(index, count, 'repetition', repetition)

        remaining = [index for index in range(len(self.row_ids)) if index not in self.stopped_token_counts]
        elapsed = self._elapsed()
        if remaining and elapsed >= GUARD_POLICY['batch_timeout_seconds']:
            for index in remaining:
                self._failure(index, count, 'batch_timeout', dict(limit_seconds=GUARD_POLICY['batch_timeout_seconds']))
            remaining = []
        if remaining:
            self._last_memory = self._memory_snapshot(input_ids.device)
            if (self._last_memory is not None and
                    self._last_memory['available_estimate_bytes'] < GUARD_POLICY['minimum_cuda_available_bytes']):
                for index in remaining:
                    self._failure(index, count, 'low_cuda_memory', dict(self._last_memory))
        all_stopped = len(self.stopped_token_counts) == len(self.row_ids)
        self._write_heartbeat(state='idle' if all_stopped else 'active',
                              force=len(self.stopped_token_counts) != ended_before)
        return torch.tensor([index in self.stopped_token_counts for index in range(len(self.row_ids))],
                            dtype=torch.bool, device=input_ids.device)

    def finish(self, state='complete'):
        if state not in ('idle', 'complete', 'failed'):
            raise ValueError('finish state must be idle, complete or failed')
        self._finished = True
        self._write_heartbeat(state=state, force=True)
