"""Sharing experience with an experience hub (see ``hub.py``).

``sync`` uploads this computer's episodes the hub has not seen yet and
downloads everyone else's new ones into the shared-experience file. Progress
is kept in ``sync_state.json`` next to the experience, so each sync only
sends and fetches what is new.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from .memory import ExperienceStore, clean_episode, episode_id

STATE_FILE = "sync_state.json"
TIMEOUT = 20  # seconds per request
BATCH = 200
MAX_PAGES = 200  # one sync downloads at most this many pages


class CloudError(RuntimeError):
    """The hub could not be reached or refused the request (message in Chinese)."""


def check_url(url: str) -> str:
    """The hub address, cleaned; raises ValueError for anything but http(s)://host[:port][/path]."""
    url = url.strip().rstrip("/")
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.query or parts.fragment:
        raise ValueError("经验中心地址应该像 https://example.com:8800 这样")
    return url


@dataclass
class SyncResult:
    pushed: int
    pulled: int
    shared_total: int
    time: str


class HubClient:
    def __init__(self, url: str, token: str, opener=urllib.request.urlopen):
        self.url = check_url(url)
        self.token = token
        self._open = opener

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.url + path, data=data, method=method, headers={
            "Authorization": f"Bearer {self.token}", "Content-Type": "application/json", "Accept": "application/json"})
        try:
            with self._open(request, timeout=TIMEOUT) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            try:
                message = json.loads(exc.read()).get("error") or exc.reason
            except (ValueError, AttributeError):
                message = exc.reason
            if exc.code == 401:
                raise CloudError("经验中心不认这个访问令牌：检查令牌有没有填错，或请管理员重新生成") from exc
            raise CloudError(f"经验中心返回 {exc.code}：{message}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise CloudError(f"连不上经验中心 {self.url}：{reason}") from exc
        except ValueError as exc:
            raise CloudError("经验中心的回复不是 JSON，地址填的可能不是经验中心") from exc

    def upload(self, episodes: list[dict]) -> dict:
        return self._request("POST", "/api/v1/episodes", {"episodes": episodes})

    def download(self, after: int) -> dict:
        return self._request("GET", f"/api/v1/episodes?after={int(after)}")

    def stats(self) -> dict:
        return self._request("GET", "/api/v1/stats")


class SyncState:
    """How far this computer has synced with one hub."""

    def __init__(self, home: Path | str):
        self.path = Path(home) / STATE_FILE
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        self.data = data if isinstance(data, dict) else {}

    def for_hub(self, url: str) -> dict:
        hub = self.data.get(url)
        return hub if isinstance(hub, dict) else {}

    def save(self, url: str, **values) -> None:
        self.data[url] = {**self.for_hub(url), **values}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)


def pending(memory: ExperienceStore, sent: int) -> list[dict]:
    """Local episodes after the first ``sent``, in shareable form (ids filled in)."""
    out = []
    for ep in memory.episodes[sent:]:
        try:
            out.append(clean_episode({**ep, "id": episode_id(ep)}))
        except ValueError:
            out.append(None)  # kept as a placeholder so the count still advances
    return out


def sync(memory: ExperienceStore, home: Path | str, client: HubClient, lock=None) -> SyncResult:
    """Upload new local episodes, then download others' new ones.

    ``lock`` (if given) guards ``memory``; network calls run outside it.
    """
    state = SyncState(home)
    hub = state.for_hub(client.url)
    sent, after = int(hub.get("sent", 0)), int(hub.get("after", 0))
    with lock or _NoLock():
        ids = [episode_id(ep) for ep in memory.episodes]
        last = hub.get("sent_id")
        if isinstance(last, str):
            # by id, not by count: if the experience file was cleared or rewritten, the count points past new
            # episodes; an id that is gone means start over (the hub ignores episodes it already has)
            sent = ids.index(last) + 1 if last in ids else 0
        sent = min(sent, len(memory.episodes))
        todo = pending(memory, sent)
    pushed = 0
    for start in range(0, len(todo), BATCH):
        batch = todo[start:start + BATCH]
        good = [ep for ep in batch if ep is not None]
        if good:
            pushed += int(client.upload(good).get("accepted", 0))
        sent += len(batch)
        state.save(client.url, sent=sent, sent_id=ids[sent - 1])
    pulled = 0
    for _ in range(MAX_PAGES):
        page = client.download(after)
        fresh = []
        for raw in page.get("episodes") or []:
            try:
                fresh.append(clean_episode(raw))
            except ValueError:
                continue  # a record this version cannot read
        with lock or _NoLock():
            pulled += memory.add_shared(fresh)
        after = int(page.get("next", after))
        state.save(client.url, after=after)
        if not page.get("more"):
            break
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state.save(client.url, last_sync=now)
    return SyncResult(pushed=pushed, pulled=pulled, shared_total=len(memory.shared), time=now)


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
