"""Experience memory: what happened every time the agent solved something.

Each solve ("episode") is appended to a JSON-lines file. From these records
the agent learns, per problem class (``kind/dim/boundary``):

* which methods succeed and how much work they need, to rank methods, and
* how fine a grid reached a given tolerance, to start near the right size
  instead of climbing up from a coarse grid every time.

Each episode also records the problem's feature tags (``stiff``, ``rough``,
``long``; see ``features.py``). Once the class has episodes with the same
tags, only those count, so a stiff problem does not inherit what worked on
gentle ones.

Episodes shared by other people through an experience hub (see ``hub.py``)
are kept in a second file and count the same as local ones.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
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


_ID = re.compile(r"[0-9a-f]{32}")
_SIGNATURE = re.compile(r"[a-z][a-z0-9_]{0,40}/[0-3]d/[a-z][a-z0-9_]{0,20}")
_NAME = re.compile(r"[A-Za-z0-9_.\-]{1,80}")
_OUTCOMES = {"success", "unstable", "error", "stagnated", "max_n", "budget"}
_TAG = re.compile(r"[a-z]{1,20}")
MAX_TRIALS = 50
EXPLORE_RATIO = 8.0  # an untried method is explored first only if predicted to cost at most this many times the cheapest
MAX_PROBLEM_CHARS = 4000


def episode_id(episode: dict) -> str:
    """The episode's id; older records without one get a stable id from their content."""
    value = episode.get("id")
    if isinstance(value, str) and _ID.fullmatch(value):
        return value
    blob = json.dumps(episode, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _number(value, low=-math.inf, high=math.inf, allow_none=True):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("not a number")
    value = float(value)
    if math.isnan(value) or not low <= value <= high:
        raise ValueError("number out of range")
    return value if math.isfinite(value) else None


def _count(value, high=10**9, allow_none=True):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= high:
        raise ValueError("not a count")
    return value


def clean_episode(episode) -> dict:
    """The fields of an episode that may be shared, checked; raises ValueError.

    Shared episodes come from other people, so only plain data with known
    shapes is kept (no code, no free text beyond the problem's expressions).
    """
    if not isinstance(episode, dict):
        raise ValueError("episode must be an object")
    signature = episode.get("signature")
    if not isinstance(signature, str) or not _SIGNATURE.fullmatch(signature):
        raise ValueError("bad signature")
    method = episode.get("method")
    if method is not None and not (isinstance(method, str) and _NAME.fullmatch(method)):
        raise ValueError("bad method name")
    error_kind = episode.get("error_kind", "exact")
    if error_kind not in ("exact", "estimated"):
        raise ValueError("bad error kind")
    trials = episode.get("methods", [])
    if not isinstance(trials, list) or len(trials) > MAX_TRIALS:
        raise ValueError("bad method list")
    clean_trials = []
    for trial in trials:
        if not isinstance(trial, dict) or not (isinstance(trial.get("method"), str) and _NAME.fullmatch(trial["method"])):
            raise ValueError("bad trial")
        if trial.get("outcome") not in _OUTCOMES:
            raise ValueError("bad trial outcome")
        clean_trials.append({
            "method": trial["method"], "outcome": trial["outcome"], "n": _count(trial.get("n"), 10**7),
            "error": _number(trial.get("error"), 0.0), "error_kind": trial.get("error_kind", error_kind)
            if trial.get("error_kind", error_kind) in ("exact", "estimated") else error_kind,
            "work": _number(trial.get("work"), 0.0),
        })
        if trial["outcome"] == "success" and clean_trials[-1]["n"] is None:
            raise ValueError("a success needs its grid size")
    problem = episode.get("problem")
    if problem is not None:
        if not isinstance(problem, dict) or len(json.dumps(problem, ensure_ascii=False)) > MAX_PROBLEM_CHARS:
            raise ValueError("bad problem")
    tags = episode.get("tags", [])
    if not isinstance(tags, list) or len(tags) > 8 or not all(isinstance(t, str) and _TAG.fullmatch(t) for t in tags):
        raise ValueError("bad tags")
    time = episode.get("time")
    if time is not None and not (isinstance(time, str) and len(time) <= 40):
        raise ValueError("bad time")
    return {
        "id": episode_id(episode),
        "time": time,
        "signature": signature,
        "problem": problem,
        "tol": _number(episode.get("tol"), 0.0, 1.0, allow_none=False),
        "success": bool(episode.get("success")),
        "method": method,
        "n": _count(episode.get("n"), 10**7),
        "error": _number(episode.get("error"), 0.0),
        "error_kind": error_kind,
        "solver_runs": _count(episode.get("solver_runs"), 10**6),
        "work": _number(episode.get("work"), 0.0),
        "runtime": _number(episode.get("runtime"), 0.0),
        "methods": clean_trials,
        "tags": sorted(set(tags)),
    }


def _read_jsonl(path: Path | None) -> list[dict]:
    episodes = []
    if path and path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        episodes.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # a torn final line from an interrupted write
    return episodes


class ExperienceStore:
    def __init__(self, path: Path | str | None = None, shared_path: Path | str | None = None):
        self.path = Path(path) if path else None
        self.shared_path = Path(shared_path) if shared_path else None
        self.episodes: list[dict] = _read_jsonl(self.path)  # this computer's own solves
        self.shared: list[dict] = []  # other people's, from the experience hub
        for ep in _read_jsonl(self.shared_path):
            try:
                self.shared.append(clean_episode(ep))
            except ValueError:
                continue
        self._ids = {episode_id(ep) for ep in self.episodes} | {ep["id"] for ep in self.shared}

    def add(self, episode: dict) -> None:
        episode = {"id": uuid.uuid4().hex, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **episode}
        self.episodes.append(episode)
        self._ids.add(episode["id"])
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(episode, ensure_ascii=False) + "\n")

    def add_shared(self, episodes: Iterable[dict]) -> int:
        """Keep other people's episodes (already checked); returns how many were new."""
        new = [ep for ep in episodes if ep["id"] not in self._ids]
        for ep in new:
            self._ids.add(ep["id"])
        self.shared.extend(new)
        if new and self.shared_path:
            self.shared_path.parent.mkdir(parents=True, exist_ok=True)
            with self.shared_path.open("a", encoding="utf-8") as fh:
                for ep in new:
                    fh.write(json.dumps(ep, ensure_ascii=False) + "\n")
        return len(new)

    def stats(self, signature: str, tags: Iterable[str] | None = None) -> dict[str, MethodStats]:
        """Per-method record for a problem class; with ``tags``, only episodes with
        the same feature tags, when there are any (otherwise the whole class)."""
        episodes = [ep for ep in self.episodes + self.shared if ep.get("signature") == signature]
        if tags is not None:
            wanted = sorted(set(tags))
            same = [ep for ep in episodes if sorted(set(ep.get("tags") or [])) == wanted]
            episodes = same or episodes
        table: dict[str, MethodStats] = {}
        for ep in episodes:
            for trial in ep.get("methods", []):
                st = table.setdefault(trial["method"], MethodStats(trial["method"]))
                st.tries += 1
                if trial.get("outcome") == "success":
                    st.successes += 1
                    st.work.append(float(trial.get("work") or 0.0))
                    st.grids.append((float(ep.get("tol", 0.0)), int(trial["n"]), trial.get("error_kind", "exact")))
        return table

    def rank(self, signature: str, candidates: Iterable[SolverSpec], explore: bool = True,
             predicted: dict[str, float] | None = None, tags: Iterable[str] | None = None) -> list[SolverSpec]:
        """Order methods by learned reliability, then by work to succeed.

        ``predicted`` is the work each method is expected to need on this very
        problem (from its features); it breaks ties instead of the class-wide
        average, and orders untried methods. With ``explore`` an untried
        method goes first (optimism in the face of uncertainty) unless its
        predicted work is more than EXPLORE_RATIO times the cheapest prediction.
        """
        stats = self.stats(signature, tags)
        candidates = list(candidates)
        predicted = {k: v for k, v in (predicted or {}).items() if v is not None}
        cheapest = min((predicted.get(s.name, math.inf) for s in candidates), default=math.inf)

        def plausible(spec):
            work = predicted.get(spec.name)
            return work is None or not math.isfinite(cheapest) or work <= EXPLORE_RATIO * cheapest

        def key(item):
            index, spec = item
            st = stats.get(spec.name)
            work = predicted.get(spec.name, math.inf)
            if st is None:
                # untried: optimistic when exploring and plausible, otherwise after known-good methods
                return (0 if explore and plausible(spec) else 1, 0.0, work, -spec.order, index)
            band = round(st.success_rate, 1)
            return (1 if explore else 0, -band, work if spec.name in predicted else st.mean_work, 0.0, index)

        return [spec for _, spec in sorted(enumerate(candidates), key=key)]

    def suggest_n(self, signature: str, spec: SolverSpec, tol: float, error_kind: str,
                  tags: Iterable[str] | None = None) -> int | None:
        """Grid size predicted to reach ``tol`` from past successes, if any.

        Each success at (tol_i, n_i) predicts n_i * (tol_i / tol)^(1/order);
        the largest prediction among recent successes is used, rounded up to
        a power of two so grids stay nested for error estimation.
        """
        st = self.stats(signature, tags).get(spec.name)
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
        for signature in sorted({ep.get("signature", "") for ep in self.episodes + self.shared}):
            for name, st in sorted(self.stats(signature).items()):
                rows.append({
                    "signature": signature,
                    "method": name,
                    "tries": st.tries,
                    "successes": st.successes,
                    "mean_work": st.mean_work,
                })
        return rows
