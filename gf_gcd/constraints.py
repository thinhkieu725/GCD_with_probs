"""Constraint modules: adapt a GF runtime to HuggingFace's `prefix_allowed_tokens_fn`.

Extracted from epfl-dlab/GCD (src/constrained_generation/{constrained,gf_constrained}.py),
decoupled from the original repo's batch/collator format. A constraint produces a
callable `fn(batch_id, sent_tokens) -> allowed token ids` that `model.generate`
invokes at every decoding step.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

PrefixAllowedTokensFn = Callable[[int, torch.Tensor], Iterable[int]]


class _SentenceCursor:
    """Recovers, from a growing token/word sequence alone, where the
    *current* grammar sentence begins.

    The compiled grammar accepts exactly one sentence, so once the model
    has generated more than that, the constraint must stop handing the
    whole "partial output so far" to the grammar and instead pass only the
    tail generated since the last sentence boundary. A step only ever
    offers tokens the grammar allows as a continuation of that tail, plus
    -- exactly when the tail has no legal continuation left (a *forced*
    stop) -- the grammar's own start-of-sentence tokens, so the LM can
    either end generation there or begin a new sentence. Restricting new
    sentences to start only right after a forced stop (never as an early
    exit from a sentence that could still continue) is what keeps "did
    this token continue the sentence or start a new one" recoverable from
    token content alone.

    That recoverability matters because there is no good place to *store*
    a cursor per batch row: HF beam search reorders rows between decoding
    steps (row `i` at step `t` need not be the same hypothesis as row `i`
    at step `t+1`), so any cursor keyed by row index would silently track
    the wrong hypothesis. Keying by the token sequence itself sidesteps
    that -- and it is cheap, because `generate` only ever grows a sequence
    that was itself handed to this cursor (one token shorter) at the
    previous step, so the recursion below is a cache hit down to the one
    new token, i.e. one grammar query per step, amortized.
    """

    def __init__(self, cache_size: int = 32768):
        self._cache: Dict[tuple, Tuple[int, object]] = {}
        self._cache_size = cache_size

    def resolve(self, tokens: tuple, pgf_name, query, is_forced_stop) -> Tuple[int, object]:
        """Return `(cursor, result)` for `tokens`.

        `cursor` is the index where the current sentence starts within
        `tokens`. `result` is `query(tokens[cursor:])` -- whatever a
        grammar continuation query returns (an allowed-token list, a
        next-token distribution, ...); returned so callers that already
        need it don't have to query again.

        `query(tail: tuple) -> R` answers a grammar continuation query for
        `tail`. `is_forced_stop(R) -> bool` says whether that answer means
        "this tail is complete, with no legal continuation left".
        """
        key = (pgf_name, tokens)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if not tokens:
            cursor = 0
        else:
            prev_cursor, prev_result = self.resolve(tokens[:-1], pgf_name, query, is_forced_stop)
            cursor = len(tokens) - 1 if is_forced_stop(prev_result) else prev_cursor
        result = query(tokens[cursor:])
        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[key] = (cursor, result)
        return cursor, result


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

    The grammar accepts exactly one sentence. Each decoding step masks
    against the tail generated since the last sentence boundary (found by
    `_SentenceCursor`, see its docstring) rather than the whole output, so
    a run can produce more than one grammar-conformant sentence: once a
    sentence has no legal continuation left, the mask offers both
    `eos_token_id` (end generation here) and the grammar's own
    start-of-sentence tokens (begin another sentence). A sentence can only
    ever end where the grammar forces it to -- this backend has no notion
    of an *optional* stop partway through an otherwise still-extendable
    sentence (see `GFProbConstraint` for that).

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
        verbose: bool = False,
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
            verbose: print per-decoding-step timing (GF server round-trip) to
                the console. Useful for eyeballing where time goes during a run.
        """
        if not os.path.isdir(grammar_dir):
            raise FileNotFoundError(f"grammar_dir does not exist: {grammar_dir}")
        self.grammar_dir = grammar_dir
        self.default_grammar = default_grammar
        self.eos_token_id = eos_token_id
        self.name = name
        self.verbose = verbose
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

        cursor = _SentenceCursor()

        def query(tail: tuple, pgf: Optional[str]) -> List[int]:
            return self.runtime.get_prefix_allowed_tokens_for_LM(tail, pgf=pgf)

        def fn(batch_id: int, sent_tokens: torch.Tensor) -> List[int]:
            sent_tokens = sent_tokens.squeeze(0) if sent_tokens.dim() > 1 else sent_tokens
            generated = tuple(sent_tokens[prompt_lengths[batch_id]:].tolist())
            pgf = grammars[batch_id] if grammars is not None else None
            t0 = time.perf_counter()
            _, allowed = cursor.resolve(
                generated,
                pgf,
                query=lambda tail: query(tail, pgf),
                is_forced_stop=lambda allowed: not allowed,
            )
            if not allowed:
                # current sentence has no legal continuation left: offer
                # ending generation here, or starting a fresh sentence
                allowed = list(query((), pgf))
                if self.eos_token_id is not None:
                    allowed = allowed + [self.eos_token_id]
            if self.verbose:
                print(
                    f"[gf-mask] batch={batch_id} step={len(generated):4d}  "
                    f"gf-server complete={(time.perf_counter() - t0) * 1000:7.1f} ms  "
                    f"allowed={len(allowed)}",
                    flush=True,
                )
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

    The grammar accepts exactly one sentence, so each row is scored against
    the tail generated since its last sentence boundary (`_SentenceCursor`),
    not the whole row. When that tail has no legal continuation left
    (`GF_EOS` is the *only* candidate `next_token_logprobs` returns), the
    fusion also scores the grammar's own start-of-sentence tokens at that
    step -- alongside `eos_token_id` -- so a beam can either end there or
    continue into another grammar-conformant sentence. An *optional* stop
    (`GF_EOS` alongside live continuations) only ever competes with those
    continuations for ending generation; it does not open a new sentence,
    since a token that is both a legal continuation and a legal sentence
    start would make "which one did the beam mean" unrecoverable from the
    row's content alone (see `_SentenceCursor`).

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
        verbose: bool = False,
    ):
        self.runtime = runtime
        self.prompt_lengths = list(prompt_lengths)
        self.eos_token_id = eos_token_id
        self.lambda_gf = lambda_gf
        self.gf_temperature = gf_temperature
        self.grammars = grammars
        self.pad_token_id = pad_token_id
        self.id_prefix = id_prefix
        self.verbose = verbose
        self._step = 0
        self._last_step_end = None  # perf_counter() timestamp when __call__ last returned
        self._cursor = _SentenceCursor()

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        from gf_gcd.gf_probs import GF_EOS

        step_start = time.perf_counter()
        # time since we last handed control back to `generate()`: covers the
        # LM forward pass for this step (+ whatever sampling/processors ran
        # before us). Not a clean isolated measurement, but a good proxy for
        # "how long did computing the LLM's logits take" at a glance.
        llm_forward_s = step_start - self._last_step_end if self._last_step_end is not None else None

        n_rows = input_ids.shape[0]
        group_size = n_rows // len(self.prompt_lengths)  # num_beams (or 1)
        fused = torch.full_like(scores, float("-inf"))

        parse_s = 0.0  # time spent in pgf `complete()` (grammar parsing + probs)
        combine_s = 0.0  # time spent fusing LM scores with GF log-probs
        n_parsed = 0
        n_finished = 0

        def to_vocab_id(token: str) -> int:
            if token == GF_EOS:
                return self.eos_token_id
            try:
                return int(token[len(self.id_prefix) :] if self.id_prefix else token)
            except ValueError as e:
                raise ValueError(
                    "GFProbLogitsProcessor requires a token-ID grammar whose "
                    f"terminals are {self.id_prefix!r} + token id (digit-only "
                    "terminals break the C runtime's complete()); the grammar "
                    f"produced {token!r}"
                ) from e

        for row in range(n_rows):
            batch_id = row // group_size
            generated = tuple(input_ids[row, self.prompt_lengths[batch_id] :].tolist())  # strips the prompt

            # a finished row (eos already emitted) only continues with eos/pad
            if self.eos_token_id in generated:
                fused[row, self.eos_token_id] = scores[row, self.eos_token_id]
                if self.pad_token_id is not None:
                    fused[row, self.pad_token_id] = scores[row, self.pad_token_id]
                n_finished += 1
                continue

            pgf_name = self.grammars[batch_id] if self.grammars is not None else None

            def query(tail: tuple, _pgf_name=pgf_name):
                return self.runtime.next_token_logprobs(
                    [f"{self.id_prefix}{t}" for t in tail],
                    pgf_name=_pgf_name,
                    temperature=self.gf_temperature,
                )

            def is_forced_stop(dist) -> bool:
                return not dist or set(dist) <= {GF_EOS}

            t_parse_start = time.perf_counter()
            _, dist = self._cursor.resolve(generated, pgf_name, query, is_forced_stop)
            t_parse_end = time.perf_counter()
            parse_s += t_parse_end - t_parse_start
            n_parsed += 1

            if not dist:
                # dead end (grammar exhausted): permit stopping
                fused[row, self.eos_token_id] = scores[row, self.eos_token_id]
                combine_s += time.perf_counter() - t_parse_end
                continue

            for token, gf_logprob in dist.items():
                idx = to_vocab_id(token)
                fused[row, idx] = scores[row, idx] + self.lambda_gf * gf_logprob

            if is_forced_stop(dist):
                # this sentence has no legal continuation left: also score
                # the grammar's own start-of-sentence tokens, so the beam
                # can continue into another grammar-conformant sentence
                # instead of only being allowed to stop here
                stop_logprob = dist[GF_EOS]
                start_dist = query(())
                for token, gf_logprob in start_dist.items():
                    if token == GF_EOS:
                        continue  # an empty sentence isn't a real continuation
                    idx = to_vocab_id(token)
                    fused[row, idx] = scores[row, idx] + self.lambda_gf * (stop_logprob + gf_logprob)
            combine_s += time.perf_counter() - t_parse_end

        step_end = time.perf_counter()

        if self.verbose:
            llm_str = f"{llm_forward_s * 1000:7.1f} ms" if llm_forward_s is not None else "    n/a   "
            print(
                f"[step {self._step:3d}] rows={n_rows:2d} (parsed={n_parsed}, finished={n_finished})  |  "
                f"LLM logits (since prev step): {llm_str}  |  "
                f"pgf complete() [parse]: {parse_s * 1000:7.1f} ms  |  "
                f"combine logits: {combine_s * 1000:6.2f} ms  |  "
                f"logits_processor total: {(step_end - step_start) * 1000:7.1f} ms",
                flush=True,
            )

        self._step += 1
        self._last_step_end = time.perf_counter()
        return fused


class GFProbConstraint(Constraint):
    """Probability-weighted grammar constraint (token-ID grammars).

    Like `GFConstraint`, but the grammar must be compiled with
    `gf -make --probs=...`; instead of masking, generation maximizes
    log P_LM + λ·log P_GF. Runs in-process on the GF C runtime (`pip
    install pgf`) — no `gf --server` processes.

    The grammar accepts exactly one sentence; see `GFProbLogitsProcessor`
    for how a run can still chain several sentences together.
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
        verbose: bool = False,
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
            verbose: print a line per decoding step with the time spent
                computing the LLM's logits (approx.), parsing/scoring via
                `pgf complete()`, and combining the two, so processing time
                per stage can be eyeballed during a run.
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
        self.verbose = verbose
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
            verbose=self.verbose,
        )

    def prefix_allowed_tokens_fn(
        self,
        prompt_lengths: Sequence[int],
        grammars: Optional[Sequence[Optional[str]]] = None,
    ) -> PrefixAllowedTokensFn:
        """Binary view of the same grammar (compatibility with mask-only use).

        Like `GFProbLogitsProcessor`, masks against the tail generated since
        the last sentence boundary rather than the whole output, and -- at a
        forced stop (no legal continuation left) -- also allows the
        grammar's own start-of-sentence tokens, so generation can chain into
        another sentence instead of only being allowed to stop. See
        `_SentenceCursor`.
        """
        from gf_gcd.gf_probs import GF_EOS

        if grammars is not None and len(grammars) != len(prompt_lengths):
            raise ValueError("grammars and prompt_lengths must have the same length")

        cursor = _SentenceCursor()

        def to_vocab_ids(dist) -> List[int]:
            ids = []
            for token in dist:
                if token == GF_EOS:
                    if self.eos_token_id is not None:
                        ids.append(self.eos_token_id)
                else:
                    ids.append(int(token[len(self.id_prefix) :] if self.id_prefix else token))
            return ids

        def is_forced_stop(dist) -> bool:
            return not dist or set(dist) <= {GF_EOS}

        def fn(batch_id: int, sent_tokens: torch.Tensor) -> List[int]:
            sent_tokens = sent_tokens.squeeze(0) if sent_tokens.dim() > 1 else sent_tokens
            generated = tuple(sent_tokens[prompt_lengths[batch_id] :].tolist())
            pgf = grammars[batch_id] if grammars is not None else None

            def query(tail: tuple, _pgf=pgf):
                return self.runtime.next_token_logprobs(
                    [f"{self.id_prefix}{t}" for t in tail], pgf_name=_pgf
                )

            t0 = time.perf_counter()
            _, dist = cursor.resolve(generated, pgf, query, is_forced_stop)
            if self.verbose:
                print(
                    f"[gf-mask] batch={batch_id} step={len(generated):4d}  "
                    f"pgf complete()={(time.perf_counter() - t0) * 1000:7.1f} ms  "
                    f"candidates={len(dist)}",
                    flush=True,
                )
            allowed = to_vocab_ids(dist)
            if is_forced_stop(dist):
                start_dist = query(())
                allowed = allowed + [
                    int(t[len(self.id_prefix) :] if self.id_prefix else t)
                    for t in start_dist
                    if t != GF_EOS
                ]
            if not allowed and self.eos_token_id is not None:
                return [self.eos_token_id]
            return allowed

        return fn
