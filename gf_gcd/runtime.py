"""GF runtime: serves compiled .pgf grammars over HTTP and answers completion queries.

Extracted from epfl-dlab/GCD (src/constrained_generation/gf_runtime.py) with the
following changes:
- real readiness polling instead of a fixed sleep after launching a server
- context-manager support so GF server processes are always cleaned up
- clearer errors when the `gf` binary or a grammar file is missing
"""

import itertools
import json
import logging
import os
import shlex
import shutil
import socket
import subprocess
import time
from typing import List, Optional, Sequence, Union
from urllib.parse import urljoin

import requests

log = logging.getLogger(__name__)


class GFRuntime:
    """Abstract GF runtime.

    `complete` takes the tokens produced so far and returns the list of tokens
    the grammar allows next.
    """

    def complete(self, input_tokens: Sequence[Union[int, str]], pgf: Optional[str] = None) -> List[str]:
        raise NotImplementedError


def _concat_tokens_to_str(tokens: Sequence[Union[int, str]]) -> str:
    # GF's `complete` command expects space-separated tokens with a trailing
    # space, meaning "the last token is finished, complete the next one".
    return " ".join(str(x) for x in tokens) + " "


def _is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return True
        except OSError:
            return False


class GFServerPoolRuntime(GFRuntime):
    """A pool of `gf --server` processes queried round-robin over HTTP."""

    DOMAIN = "http://localhost"
    READINESS_TIMEOUT_S = 10.0

    def __init__(
        self,
        default_pgf: Optional[str],
        grammar_dir: str,
        port_start: int = 41296,
        pool_size: int = 2,
        gf_binary: str = "gf",
    ):
        if shutil.which(gf_binary) is None:
            raise RuntimeError(
                f"GF binary {gf_binary!r} not found on PATH. Install Grammatical Framework "
                "(https://www.grammaticalframework.org/download/) or pass `gf_binary`."
            )
        if not os.path.isdir(grammar_dir):
            raise FileNotFoundError(f"grammar_dir does not exist: {grammar_dir}")

        self.grammar_dir = grammar_dir
        self._default_pgf = default_pgf
        self._gf_binary = gf_binary
        self.server_processes: List[subprocess.Popen] = []
        self.urls: List[str] = []

        for _ in range(pool_size):
            port = self._find_free_port(port_start)
            url = f"{self.DOMAIN}:{port}"
            self.server_processes.append(self._launch_server(port))
            self._wait_until_ready(url)
            self.urls.append(url)
            port_start = port + 1

        self._server_iterator = itertools.cycle(self.urls)

    # ---------------------------------------------------------------- lifecycle

    def _find_free_port(self, port: int) -> int:
        while not _is_port_available(port):
            port += 1
        return port

    def _launch_server(self, port: int) -> subprocess.Popen:
        cmd = f"{self._gf_binary} --document-root={self.grammar_dir} --server={port}"
        log.debug("Launching GF server: %s", cmd)
        return subprocess.Popen(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            close_fds=True,
        )

    def _wait_until_ready(self, url: str) -> None:
        deadline = time.monotonic() + self.READINESS_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                requests.get(url, timeout=1)
                return
            except requests.ConnectionError:
                time.sleep(0.05)
        raise RuntimeError(f"GF server at {url} did not become ready within {self.READINESS_TIMEOUT_S}s")

    def clear(self) -> None:
        """Terminate all GF server processes."""
        for process in self.server_processes:
            process.terminate()
        for process in self.server_processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        self.server_processes = []

    def __enter__(self) -> "GFServerPoolRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.clear()

    def __del__(self):
        try:
            self.clear()
        except Exception:
            pass

    # ---------------------------------------------------------------- queries

    def _resolve_pgf(self, pgf: Optional[str]) -> str:
        if pgf is not None and os.path.exists(os.path.join(self.grammar_dir, pgf)):
            return pgf
        if self._default_pgf is None:
            raise RuntimeError(f"Grammar {pgf!r} not found in {self.grammar_dir} and no default_pgf is set")
        log.warning(f"Grammar {pgf!r} not found in {self.grammar_dir}. Falling back to default PGF: {self._default_pgf}")
        return self._default_pgf

    def complete(self, input_tokens: Sequence[Union[int, str]], pgf: Optional[str] = None) -> List[str]:
        """Return the tokens the grammar allows after `input_tokens`."""
        url = urljoin(next(self._server_iterator) + "/", self._resolve_pgf(pgf))
        params = {"command": "complete", "input": _concat_tokens_to_str(input_tokens)}

        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise RuntimeError(
                f"GF server returned status {response.status_code} for {url}: {response.text[:500]}"
            )

        parsed = json.loads(response.text)
        assert len(parsed) == 1, f"expected a single completion record, got {len(parsed)}"
        return parsed[0]["completions"]


class GFServerRuntimeForLM(GFServerPoolRuntime):
    """GF runtime for grammars whose terminals are LM token IDs (as strings)."""

    def get_prefix_allowed_tokens_for_LM(
        self,
        prefix_tokens: Union[Sequence[int], "torch.Tensor"],  # noqa: F821
        pgf: Optional[str] = None,
    ) -> List[int]:
        if hasattr(prefix_tokens, "tolist"):  # torch.Tensor / np.ndarray
            prefix_tokens = prefix_tokens.tolist()
        completions = self.complete([str(x) for x in prefix_tokens], pgf=pgf)
        try:
            return [int(x) for x in completions]
        except ValueError as e:
            raise ValueError(
                "Grammar returned non-integer completions — token-ID grammars must have "
                f"terminals that are token IDs. Got: {completions[:10]}"
            ) from e
