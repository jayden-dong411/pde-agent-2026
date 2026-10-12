"""Experience memory: what happened every time the agent solved something.

Each solve ("episode") is appended to a JSON-lines file. From these records
the agent learns, per problem class (``kind/dim/boundary``):

* which methods succeed and how much work they need, to rank methods, and
* how fine a grid reached a given tolerance, to start near the right size
  instead of climbing up from a coarse grid every time.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .solvers.base import SolverSpec


@dataclass
class MethodStats:
    method: str
    tries: int = 0
    successes: int = 0
    work: list[float] = field(default_factory=list)  # work spent in successful episodes
    grids: list[tuple[float, int, str]] = field(default_factory=list)  # (tol, n, error kind) on success

    @property
    def failures(self) -> int:
        return self.tries - self.successes

    @property
    def success_rate(self) -> float:
        """Posterior mean with a uniform prior, so one lucky run is not 100%."""
        return (self.successes + 1) / (self.tries + 2)

    @property
    def mean_work(self) -> float:
        return sum(self.work) / len(self.work) if self.work else math.inf


class ExperienceStore:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else None
        self.episodes: list[dict] = []
        if self.path and self.path.exists():
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            self.episodes.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue  # a torn final line from an interrupted write

    def add(self, episode: dict) -> None:
        episode = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **episode}
        self.episodes.append(episode)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(episode, ensure_ascii=False) + "\n")

    def stats(self, signature: str) -> dict[str, MethodStats]:
        table: dict[str, MethodStats] = {}
        for ep in self.episodes:
            if ep.get("signature") != signature:
                continue
            for trial in ep.get("methods", []):
                st = table.setdefault(trial["method"], MethodStats(trial["method"]))
                st.tries += 1
                if trial.get("outcome") == "success":
                    st.successes += 1
                    st.work.append(float(trial.get("work", 0.0)))
                    st.grids.append((float(ep.get("tol", 0.0)), int(trial["n"]), trial.get("error_kind", "exact")))
        return table

    def rank(self, signature: str, candidates: Iterable[SolverSpec], explore: bool = True) -> list[SolverSpec]:
        """Order methods by learned reliability, then by work to succeed.

        With ``explore`` an untried method goes first (optimism in the face of
        uncertainty), so every method gets a fair trial once per problem class.
        """
        stats = self.stats(signature)
        candidates = list(candidates)

        def key(item):
            index, spec = item
            st = stats.get(spec.name)
            if st is None:
                # untried: optimistic when exploring, otherwise after known-good methods
                return (0 if explore else 1, 0.0, -spec.order, 0.0, index)
            band = round(st.success_rate, 1)
            return (1 if explore else 0, -band, 0.0, st.mean_work, index)

        return [spec for _, spec in sorted(enumerate(candidates), key=key)]

    def suggest_n(self, signature: str, spec: SolverSpec, tol: float, error_kind: str) -> int | None:
        """Grid size predicted to reach ``tol`` from past successes, if any.

        Each success at (tol_i, n_i) predicts n_i * (tol_i / tol)^(1/order);
        the largest prediction among recent successes is used, rounded up to
        a power of two so grids stay nested for error estimation.
        """
        st = self.stats(signature).get(spec.name)
        if not st or not st.grids:
            return None
        order = max(spec.order, 0.5)
        recent = [(t, n) for t, n, kind in st.grids[-5:] if kind == error_kind and t > 0] or [
            (t, n) for t, n, _ in st.grids[-5:] if t > 0
        ]
        if not recent:
            return None
        predicted = max(n * (t / tol) ** (1.0 / order) for t, n in recent)
        return max(4, 2 ** math.ceil(math.log2(max(predicted, 1.0)) - 1e-9))

    def summary(self) -> list[dict]:
        rows = []
        for signature in sorted({ep.get("signature", "") for ep in self.episodes}):
            for name, st in sorted(self.stats(signature).items()):
                rows.append({
                    "signature": signature,
                    "method": name,
                    "tries": st.tries,
                    "successes": st.successes,
                    "mean_work": st.mean_work,
                })
        return rows
