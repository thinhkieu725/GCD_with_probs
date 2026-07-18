"""Plain constrained-generation wrapper around a HuggingFace causal LM.

Distilled from epfl-dlab/GCD's HFModelPL.sample(), with the PyTorch Lightning,
Hydra and wandb scaffolding removed. Adds optional per-token log-probabilities
via `transformers`' compute_transition_scores.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gf_gcd.constraints import Constraint

log = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """One entry per prompt; `texts[i]` holds the num_return_sequences candidates."""

    texts: List[List[str]]
    sequences: torch.Tensor  # (batch * num_return_sequences, gen_len), prompt removed
    # per generated token: log P(token | prefix); same leading dim as `sequences`.
    token_logprobs: Optional[torch.Tensor] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def first_texts(self) -> List[str]:
        return [candidates[0] for candidates in self.texts]


class ConstrainedGenerator:
    """Loads a HF causal LM + tokenizer and generates with an optional grammar constraint."""

    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, Path],
        half_precision: bool = False,
        device_map: str = "auto",
        hf_model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        model_path = pretrained_model_name_or_path
        if isinstance(model_path, str) and (model_path.startswith("/") or model_path.startswith(".")):
            model_path = Path(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # left padding so that every prompt in a batch ends at the same position
        # and prompt_length cleanly separates prompt from generation
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if half_precision else torch.float32,
            device_map=device_map,
            **(hf_model_kwargs or {}),
        )
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompts: Union[str, Sequence[str]],
        constraint: Optional[Constraint] = None,
        grammars: Optional[Sequence[Optional[str]]] = None,
        return_token_logprobs: bool = False,
        skip_special_tokens: bool = True,
        seed: Optional[int] = None,
        **hf_generation_params: Any,
    ) -> GenerationResult:
        """Generate completions, optionally constrained by a grammar.

        Args:
            prompts: a prompt or a batch of prompts.
            constraint: a `Constraint`; None means unconstrained generation.
            grammars: per-prompt .pgf filename (requires `constraint`); None
                entries fall back to the constraint's default grammar.
            return_token_logprobs: also return log P of each generated token.
            hf_generation_params: forwarded to `model.generate`
                (max_new_tokens, num_beams, do_sample, ...).
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if grammars is not None and constraint is None:
            raise ValueError("`grammars` was given but no `constraint`")

        if seed is not None:
            torch.manual_seed(seed)

        input_device = next(self.model.parameters()).device
        encoded = self.tokenizer(list(prompts), padding=True, return_tensors="pt").to(input_device)
        prompt_length = encoded["input_ids"].shape[1]

        prefix_allowed_tokens_fn = None
        logits_processor = None
        if constraint is not None:
            prompt_lengths = [prompt_length] * len(prompts)
            # constraints that carry probabilities provide a logits processor
            # (score reweighting); others fall back to binary masking
            processor = constraint.logits_processor(prompt_lengths=prompt_lengths, grammars=grammars)
            if processor is not None:
                from transformers import LogitsProcessorList

                logits_processor = LogitsProcessorList([processor])
            else:
                prefix_allowed_tokens_fn = constraint.prefix_allowed_tokens_fn(
                    prompt_lengths=prompt_lengths,
                    grammars=grammars,
                )

        hf_generation_params.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        outputs = self.model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            return_dict_in_generate=True,
            output_scores=return_token_logprobs,
            **hf_generation_params,
        )

        gen_sequences = outputs.sequences[:, prompt_length:]

        token_logprobs = None
        if return_token_logprobs:
            token_logprobs = self.model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                beam_indices=getattr(outputs, "beam_indices", None),
                normalize_logits=True,
            )

        decoded = self.tokenizer.batch_decode(gen_sequences, skip_special_tokens=skip_special_tokens)
        num_return = len(decoded) // len(prompts)
        grouped = [decoded[i * num_return : (i + 1) * num_return] for i in range(len(prompts))]

        return GenerationResult(
            texts=grouped,
            sequences=gen_sequences,
            token_logprobs=token_logprobs,
            raw={"generation_outputs": outputs},
        )
