"""Qwen3.5-4B + LoRA scorer: exactly the prompt and scoring used in the thesis experiments.

System turn = the dataset's instruction; user turn = its text prefix + text (cut to 128 tokens); thinking disabled
(the chat template adds an empty <think></think> block); one forward pass; softmax over the dataset's answer-word
tokens at the last position. Left padding, batches of 8, explicit position ids (as at training-time evaluation).
One base model serves every dataset; adapters are loaded once and switched with set_adapter.
"""
import numpy as np

from common import DEFAULT_MODEL_DIR, MAX_TEXT_TOKENS, domain, pick_device, pick_dtype


class Scorer:
    def __init__(self, model_dir=None, device='auto', dtype=None, threads=None):
        """device: auto (cuda, else Apple-silicon mps, else cpu), cuda, mps or cpu. dtype: None = bfloat16 on cuda and
        cpu; on mps bfloat16 when supported, else float32."""
        self.device = pick_device(device)
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM
        self.torch = torch
        model_dir = model_dir or DEFAULT_MODEL_DIR
        if not (model_dir / 'config.json').is_file():
            raise SystemExit(f'base model not found in {model_dir} (it ships in models/Qwen3.5-4B; see README)')
        if threads:
            torch.set_num_threads(threads)
        dtype = dtype or pick_dtype(self.device)
        self.dtype = dtype
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.tok.padding_side = 'left'
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.base = Qwen3_5ForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=getattr(torch, dtype),
                                                       attn_implementation='sdpa', device_map={'': self.device}).eval()
        self.model, self.loaded, self.active, self.count, self.codes = None, {}, None, 0, {}

    def answer_codes(self, words):
        """Token ids of the answer words; each must be one distinct token."""
        key = tuple(words)
        if key not in self.codes:
            codes = [self.tok.encode(w, add_special_tokens=False) for w in words]
            if not all(len(c) == 1 for c in codes) or len({c[0] for c in codes}) != len(codes):
                raise RuntimeError('each answer word must be a distinct single token')
            self.codes[key] = self.torch.tensor([c[0] for c in codes], device=self.device)
        return self.codes[key]

    def use(self, adapter_dir, name=None):
        """Load an adapter once (kept in memory under `name`) and make it the active one."""
        from peft import PeftModel
        name = name or str(adapter_dir)
        if name not in self.loaded:
            self.count += 1
            key = f'a{self.count}'
            if self.model is None:
                self.model = PeftModel.from_pretrained(self.base, str(adapter_dir), adapter_name=key).eval()
            else:
                self.model.load_adapter(str(adapter_dir), adapter_name=key)
            self.loaded[name] = key
        self.model.set_adapter(self.loaded[name])
        self.active = name

    def drop(self, name):
        """Free a loaded adapter that is not the active one."""
        if name in self.loaded and name != self.active:
            self.model.delete_adapter(self.loaded.pop(name))

    def prompt(self, text, dom):
        d = domain(dom) if isinstance(dom, str) else dom
        ids = self.tok.encode(text, add_special_tokens=False)
        visible = self.tok.decode(ids[:MAX_TEXT_TOKENS], skip_special_tokens=True) if len(ids) > MAX_TEXT_TOKENS else text
        rendered = self.tok.apply_chat_template([{'role': 'system', 'content': d.instruction},
                                                {'role': 'user', 'content': d.prefix + visible}],
                                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if not rendered.endswith('</think>\n\n'):
            raise RuntimeError('non-thinking prompt must end with the empty thinking block')
        return self.tok.encode(rendered, add_special_tokens=False)

    def score(self, texts, dom, batch_size=8, progress=False):
        """Class probabilities of the active adapter, shape (len(texts), number of classes of the domain)."""
        torch = self.torch
        d = domain(dom) if isinstance(dom, str) else dom
        if self.model is None:
            raise RuntimeError('call use(adapter_dir) first')
        codes = self.answer_codes(d.words)
        prompts = [self.prompt(t, d) for t in texts]
        P = []
        with torch.inference_mode():
            for s in range(0, len(prompts), batch_size):
                batch = self.tok.pad({'input_ids': prompts[s:s + batch_size]}, padding=True, return_tensors='pt').to(self.device)
                pos = (batch['attention_mask'].cumsum(-1) - 1).clamp_min(0)
                logits = self.model(**batch, position_ids=pos, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
                if not torch.isfinite(logits).all():
                    raise RuntimeError('non-finite logits')
                P.append(logits[:, codes].softmax(-1).cpu().numpy())
                if progress and (s // batch_size) % 50 == 0:
                    print(f'  scored {min(s + batch_size, len(prompts))}/{len(prompts)}', flush=True)
        return np.concatenate(P) if P else np.zeros((0, d.k))
