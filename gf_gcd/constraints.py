"""Constraint modules: adapt a GF runtime to HuggingFace's `prefix_allowed_tokens_fn`.

Extracted from epfl-dlab/GCD (src/constrained_generation/{constrained,gf_constrained}.py),
decoupled from the original repo's batch/collator format. A constraint produces a
callable `fn(batch_id, sent_tokens) -> allowed token ids` that `model.generate`
invokes at every decoding step.
"""

import os
from abc import ABC, abstractmethod
from typing import Callable, Iterable, List, Optional, Sequence

import torch

PrefixAllowedTokensFn = Callable[[int, torch.Tensor], Iterable[int]]


class Constraint(ABC):
    """Interface for constraint modules.

    Implementations for other constraint sources (tries, regex, JSON schema)
    or other backends should conform to this interface.
    """

    @abstractmethod
    def prefix_allowed_tokens_fn(
        self,
        prompt_lengths: Sequence[int],
        grammars: Optional[Sequence[Optional[str]]] = None,
    ) -> PrefixAllowedTokensFn:
        """Build the per-step callback for one batch.

        Args:
            prompt_lengths: number of prompt tokens for each batch element;
                the constraint only sees tokens generated after the prompt.
            grammars: optional per-element grammar identifier (e.g. a .pgf
                filename). Elements may be None to use the default grammar.
        """
        raise NotImplementedError

    def logits_processor(
        self,
        prompt_lengths: Sequence[int],
        grammars: Optional[Sequence[Optional[str]]] = None,
    ) -> Optional["transformers.LogitsProcessor"]:  # noqa: F821
        """Optional richer interface: a processor that reweights (not just
        masks) the logits. Constraints that carry probabilities implement
        this; `ConstrainedGenerator` prefers it over the binary
        `prefix_allowed_tokens_fn` when it returns non-None."""
        return None


class GFConstraint(Constraint):
    """Constrain generation with compiled GF grammars whose terminals are token IDs.

    Owns a `GFServerRuntimeForLM` (created lazily on first use so that the
    constraint can be constructed before deciding to spend the server startup).
    """

    def __init__(
        self,
        grammar_dir: str,
        default_grammar: Optional[str] = None,
        eos_token_id: Optional[int] = None,
        pool_size: int = 2,
        port_start: int = 41296,
        gf_binary: str = "gf",
        name: Optional[str] = None,
    ):
        """
        Args:
            grammar_dir: directory containing compiled .pgf grammars.
            default_grammar: .pgf filename used when no per-sample grammar is given.
            eos_token_id: if set, returned as the only allowed token when the
                grammar has no completions left (i.e. the derivation is finished).
                Without it, a grammar that does not itself end with the EOS token
                id will crash generation once the derivation completes.
            name: optional identifier, useful for experiment logging.
        """
        if not os.path.isdir(grammar_dir):
            raise FileNotFoundError(f"grammar_dir does not exist: {grammar_dir}")
        self.grammar_dir = grammar_dir
        self.default_grammar = default_grammar
        self.eos_token_id = eos_token_id
        self.name = name
        self._runtime_kwargs = dict(pool_size=pool_size, port_start=port_start, gf_binary=gf_binary)
        self._runtime = None

    @property
    def runtime(self):
        if self._runtime is None:
            from gf_gcd.runtime import GFServerRuntimeForLM

            self._runtime = GFServerRuntimeForLM(
                default_pgf=self.default_grammar,
                grammar_dir=self.grammar_dir,
                **self._runtime_kwargs,
            )
        return self._runtime

    def prefix_allowed_tokens_fn(
        self,
        prompt_lengths: Sequence[int],
        grammars: Optional[Sequence[Optional[str]]] = None,
    ) -> PrefixAllowedTokensFn:
        if grammars is not None and len(grammars) != len(prompt_lengths):
            raise ValueError("grammars and prompt_lengths must have the same length")

        def fn(batch_id: int, sent_tokens: torch.Tensor) -> List[int]:
            sent_tokens = sent_tokens.squeeze(0) if sent_tokens.dim() > 1 else sent_tokens
            generated = sent_tokens[prompt_lengths[batch_id]:]
            pgf = grammars[batch_id] if grammars is not None else None
            allowed = self.runtime.get_prefix_allowed_tokens_for_LM(generated, pgf=pgf)
            if not allowed and self.eos_token_id is not None:
                return [self.eos_token_id]
            return allowed

        return fn

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.clear()
            self._runtime = None

    def __enter__(self) -> "GFConstraint":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class GFProbLogitsProcessor:
    """Fuse LM scores with grammar probabilities:

        fused[t] = scores[t] + lambda_gf * ln P_GF(t | prefix)   if t allowed
                   -inf                                          otherwise

    In HF beam search, `scores` arrives as log-softmaxed LM log-probs, so the
    fused quantity is exactly log P_LM + λ·log P_GF (product of experts).
    In greedy/sampling the subsequent softmax renormalizes the same fusion.
    lambda_gf = 0 recovers plain grammar masking (binary GCD).

    Conforms to `transformers.LogitsProcessor.__call__`; kept dependency-light
    by duck typing instead of subclassing.
    """

    def __init__(
        self,
        runtime,  # GFProbRuntime over a token-ID grammar
        prompt_lengths: Sequence[int],
        eos_token_id: int,
        lambda_gf: float = 0.3,
        gf_temperature: float = 1.0,
        grammars: Optional[Sequence[Optional[str]]] = None,
        pad_token_id: Optional[int] = None,
        id_prefix: str = "t",
    ):
        self.runtime = runtime
        self.prompt_lengths = list(prompt_lengths)
        self.eos_token_id = eos_token_id
        self.lambda_gf = lambda_gf
        self.gf_temperature = gf_temperature
        self.grammars = grammars
        self.pad_token_id = pad_token_id
        self.id_prefix = id_prefix

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        from gf_gcd.gf_probs import GF_EOS

        n_rows = input_ids.shape[0]
        group_size = n_rows // len(self.prompt_lengths)  # num_beams (or 1)
        fused = torch.full_like(scores, float("-inf"))

        for row in range(n_rows):
            batch_id = row // group_size
            generated = input_ids[row, self.prompt_lengths[batch_id] :].tolist() # strips the prompt

            # a finished row (eos already emitted) only continues with eos/pad
            if self.eos_token_id in generated:
                fused[row, self.eos_token_id] = scores[row, self.eos_token_id]
                if self.pad_token_id is not None:
                    fused[row, self.pad_token_id] = scores[row, self.pad_token_id]
                continue

            pgf_name = self.grammars[batch_id] if self.grammars is not None else None
            dist = self.runtime.next_token_logprobs(
                [f"{self.id_prefix}{t}" for t in generated],
                pgf_name=pgf_name,
                temperature=self.gf_temperature,
            )
            if not dist:
                # dead end (grammar exhausted): permit stopping
                fused[row, self.eos_token_id] = scores[row, self.eos_token_id]
                continue

            for token, gf_logprob in dist.items():
                if token == GF_EOS:
                    idx = self.eos_token_id
                else:
                    try:
                        idx = int(token[len(self.id_prefix) :] if self.id_prefix else token)
                    except ValueError as e:
                        raise ValueError(
                            "GFProbLogitsProcessor requires a token-ID grammar whose "
                            f"terminals are {self.id_prefix!r} + token id (digit-only "
                            "terminals break the C runtime's complete()); the grammar "
                            f"produced {token!r}"
                        ) from e
                fused[row, idx] = scores[row, idx] + self.lambda_gf * gf_logprob

        return fused


class GFProbConstraint(Constraint):
    """Probability-weighted grammar constraint (token-ID grammars).

    Like `GFConstraint`, but the grammar must be compiled with
    `gf -make --probs=...`; instead of masking, generation maximizes
    log P_LM + λ·log P_GF. Runs in-process on the GF C runtime (`pip
    install pgf`) — no `gf --server` processes.
    """

    def __init__(
        self,
        grammar_dir: Optional[str] = None,
        default_grammar: Optional[str] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        lambda_gf: float = 0.3,
        gf_temperature: float = 1.0,
        id_prefix: str = "t",
        concrete: Optional[str] = None,
        name: Optional[str] = None,
        runtime: Optional["GFProbRuntime"] = None,  # noqa: F821
    ):
        """
        Args:
            lambda_gf: weight of the grammar log-probability in the fused
                score. 0 = grammar as pure mask; 1 = full product of experts;
                larger values trust the grammar's distribution more.
            gf_temperature: temperature applied to the grammar distribution
                before fusion (T -> inf flattens it towards a pure mask).
            eos_token_id: token id that realizes GF_EOS ("the derivation may
                stop here"). Required for generation to terminate.
            id_prefix: marker prepended to token IDs in the grammar's
                terminals (see `ids_to_gf_lin`); must match how the grammar
                was tokenized. Digit-only terminals break the C runtime's
                complete(), hence the non-numeric default "t".
            runtime: an existing `GFProbRuntime` to reuse instead of building
                one from `grammar_dir`/`default_grammar`/`concrete`. The GF
                side of the fused score doesn't depend on `lambda_gf`, so
                sharing one runtime (and its completion cache) across
                several constraints that only differ in `lambda_gf` (e.g. a
                λ sweep over the same prompt) avoids reloading the .pgf and
                recomputing completions from scratch for each one.
                `grammar_dir` may be omitted when `runtime` is given.
        """
        if runtime is None and grammar_dir is None:
            raise ValueError("either `grammar_dir` or `runtime` must be given")
        if runtime is None and not os.path.isdir(grammar_dir):
            raise FileNotFoundError(f"grammar_dir does not exist: {grammar_dir}")
        self.grammar_dir = grammar_dir
        self.default_grammar = default_grammar
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.lambda_gf = lambda_gf
        self.gf_temperature = gf_temperature
        self.id_prefix = id_prefix
        self.concrete = concrete
        self.name = name
        self._runtime = runtime

    @property
    def runtime(self):
        if self._runtime is None:
            from gf_gcd.gf_probs import GFProbRuntime

            self._runtime = GFProbRuntime(
                grammar_dir=self.grammar_dir,
                default_pgf=self.default_grammar,
                concrete=self.concrete,
            )
        return self._runtime

    def logits_processor(
        self,
        prompt_lengths: Sequence[int],
        grammars: Optional[Sequence[Optional[str]]] = None,
    ) -> GFProbLogitsProcessor:
        if self.eos_token_id is None:
            raise ValueError("GFProbConstraint needs eos_token_id to terminate generation")
        if grammars is not None and len(grammars) != len(prompt_lengths):
            raise ValueError("grammars and prompt_lengths must have the same length")
        return GFProbLogitsProcessor(
            runtime=self.runtime,
            prompt_lengths=prompt_lengths,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            lambda_gf=self.lambda_gf,
            gf_temperature=self.gf_temperature,
            grammars=grammars,
            id_prefix=self.id_prefix,
        )

    def prefix_allowed_tokens_fn(
        self,
        prompt_lengths: Sequence[int],
        grammars: Optional[Sequence[Optional[str]]] = None,
    ) -> PrefixAllowedTokensFn:
        """Binary view of the same grammar (compatibility with mask-only use)."""
        from gf_gcd.gf_probs import GF_EOS

        def fn(batch_id: int, sent_tokens: torch.Tensor) -> List[int]:
            sent_tokens = sent_tokens.squeeze(0) if sent_tokens.dim() > 1 else sent_tokens
            generated = sent_tokens[prompt_lengths[batch_id] :].tolist()
            pgf = grammars[batch_id] if grammars is not None else None
            dist = self.runtime.next_token_logprobs(
                [f"{self.id_prefix}{t}" for t in generated], pgf_name=pgf
            )
            allowed = []
            for token in dist:
                if token == GF_EOS:
                    if self.eos_token_id is not None:
                        allowed.append(self.eos_token_id)
                else:
                    allowed.append(int(token[len(self.id_prefix) :] if self.id_prefix else token))
            if not allowed and self.eos_token_id is not None:
                return [self.eos_token_id]
            return allowed

        return fn
