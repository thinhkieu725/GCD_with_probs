# Device setup: CUDA, CPU-only, and Apple Silicon (mps)

`examples/full_pipeline.py` and `gf_gcd/generator.py` are currently wired for
a single CUDA GPU with a 4-bit-quantized 7-8B model (developed against a
12GB-VRAM RTX 5070). This is the reference for what to change to run on a
different device.

## Where the device lives

Two places, both in the CONFIG block of `examples/full_pipeline.py` (lines
~58-70):

```python
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE_MAP = "cuda"       # single-GPU placement; requires `accelerate`
HALF_PRECISION = True     # float16 for the non-quantized layers (embeddings, norms)
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)  # requires `bitsandbytes`; keeps a 7-8B model comfortably under 12GB VRAM
```

...which feed straight into `ConstrainedGenerator` in `main()`:

```python
generator = ConstrainedGenerator(
    MODEL_NAME,
    half_precision=HALF_PRECISION,
    device_map=DEVICE_MAP,
    hf_model_kwargs={"quantization_config": QUANTIZATION_CONFIG},
)
```

`ConstrainedGenerator.__init__` (`gf_gcd/generator.py`, ~line 39-63) does
nothing device-specific beyond passing `device_map` and `dtype` straight to
`AutoModelForCausalLM.from_pretrained`, plus splatting `hf_model_kwargs`
(which is where `quantization_config` rides in). That function is
device-agnostic — everything below is CONFIG-block changes in
`full_pipeline.py`, not code changes in `generator.py`.

The other example scripts (`quickstart.py`, `quickstart_probs.py`,
`rgl_token_level.py`) hardcode `device_map=None` (CPU) with plain `gpt2` and
don't need any of this.

## CPU-only (no CUDA, no mps)

`bitsandbytes` quantization is CUDA-only — there is no 4-bit path here, so
this also means picking a small enough model to run unquantized.

```python
MODEL_NAME = "gpt2"        # or another small model that fits in RAM unquantized
DEVICE_MAP = None          # plain CPU placement, no `accelerate` needed
HALF_PRECISION = False     # float16 has poor/no support on most CPU kernels
QUANTIZATION_CONFIG = None
```

And drop the `hf_model_kwargs` arg entirely (or pass `{}`) since there's no
quantization config to attach:

```python
generator = ConstrainedGenerator(MODEL_NAME, half_precision=HALF_PRECISION, device_map=DEVICE_MAP)
```

## Apple Silicon (mps)

Same constraint as CPU: `bitsandbytes` doesn't support `mps`, so no 4-bit
quantization — pick a model small enough to fit unquantized in unified
memory (a 3-4B instruct model is a reasonable ceiling on 16GB unified memory;
scale up with more RAM).

```python
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"   # or similar, small enough to fit in RAM unquantized
DEVICE_MAP = "mps"
HALF_PRECISION = True      # float16 works fine on mps and halves memory
QUANTIZATION_CONFIG = None
```

```python
generator = ConstrainedGenerator(MODEL_NAME, half_precision=HALF_PRECISION, device_map=DEVICE_MAP)
```

Note `torch.backends.mps.is_available()` must be `True` (macOS 12.3+, an
`mps`-built torch) — no extra `pip install` beyond base `torch` is needed,
unlike the CUDA path's `accelerate`/`bitsandbytes`.

## Picking automatically instead of hardcoding

If you'd rather not hand-edit the CONFIG block per machine, replace the
`DEVICE_MAP` line with:

```python
import torch

if torch.cuda.is_available():
    DEVICE_MAP = "cuda"
elif torch.backends.mps.is_available():
    DEVICE_MAP = "mps"
else:
    DEVICE_MAP = None
```

...and gate `QUANTIZATION_CONFIG`/`HALF_PRECISION`/`MODEL_NAME` on the same
check, since the right model size and whether 4-bit is even possible both
depend on which branch you land in.

## Dependencies

- `accelerate` — needed whenever `device_map` is anything other than `None`
  (i.e. CUDA or mps placement).
- `bitsandbytes` — needed only for `QUANTIZATION_CONFIG` (CUDA only).

Both are optional extras, not core dependencies (see `pyproject.toml`'s
`gpu` extra / `requirements.txt`), since the CPU-only path needs neither:

```bash
pip install -e ".[gpu]"       # or: pip install accelerate bitsandbytes
```
