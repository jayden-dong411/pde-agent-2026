"""Experience hub: a small server where several people's agents share experience.

Every member's agent uploads its solve records ("episodes") and downloads
everyone else's, so method choices and grid sizes learned by one person help
all of them. Only checked, plain-data records are shared (``clean_episode``);
solver code learned by one agent is never sent to another.

Run it on a machine everyone can reach::

    pde-agent hub add-member 小明          # prints 小明's access token
    pde-agent hub serve --host 0.0.0.0 --port 8800

Data lives in one folder (``--data``, default ``~/.pde_agent_hub``): a SQLite
database of episodes and ``members.json`` with the members' token hashes.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import __version__
from .memory import clean_episode

MAX_BATCH = 200  # episodes per upload
MAX_BODY = 2_000_000  # bytes per request
PAGE = 500  # episodes per download page
_MEMBER = re.compile(r"[^\s/\\]{1,40}")


def default_hub_dir() -> Path:
    return Path(os.environ.get("PDE_AGENT_HUB") or Path.home() / ".pde_agent_hub")


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class Hub:
    """The hub's storage: episodes in SQLite, members as token hashes."""

    def __init__(self, data: Path | str):
        self.dir = Path(data)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.members_path = self.dir / "members.json"
        self.lock = threading.Lock()
        self.db = sqlite3.connect(self.dir / "hub.sqlite3", check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS episodes (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            id TEXT UNIQUE NOT NULL,
            member TEXT NOT NULL,
            received TEXT NOT NULL,
            body TEXT NOT NULL)""")
        self.db.commit()

    # members -----------------------------------------------------------------
    def members(self) -> dict[str, str]:
        """Member name -> token hash."""
        try:
            data = json.loads(self.members_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)} if isinstance(data, dict) else {}

    def _save_members(self, members: dict[str, str]) -> None:
        tmp = self.members_path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(members, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self.members_path)

    def add_member(self, name: str) -> str:
        """Create (or replace) a member's token; returns the token, shown only once."""
        if not _MEMBER.fullmatch(name):
            raise ValueError("成员名不能有空格或斜杠，最多 40 个字")
        token = "pdeh_" + secrets.token_urlsafe(24)
        with self.lock:
            members = self.members()
            members[name] = _hash(token)
            self._save_members(members)
        return token

    def remove_member(self, name: str) -> bool:
        with self.lock:
            members = self.members()
            if members.pop(name, None) is None:
                return False
            self._save_members(members)
        return True

    def member_for(self, token: str) -> str | None:
        digest = _hash(token)
        for name, stored in self.members().items():
            if hmac.compare_digest(stored, digest):
                return name
        return None

    # episodes ----------------------------------------------------------------
    def upload(self, member: str, episodes: list) -> dict:
        accepted = duplicates = 0
        rejected = []
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.lock:
            for index, raw in enumerate(episodes):
                try:
                    episode = clean_episode(raw)
                except ValueError as exc:
                    rejected.append({"index": index, "reason": str(exc)})
                    continue
                cursor = self.db.execute(
                    "INSERT OR IGNORE INTO episodes (id, member, received, body) VALUES (?, ?, ?, ?)",
                    (episode["id"], member, now, json.dumps(episode, ensure_ascii=False)))
                if cursor.rowcount:
                    accepted += 1
                else:
                    duplicates += 1
            self.db.commit()
        return {"accepted": accepted, "duplicates": duplicates, "rejected": rejected}

    def download(self, after: int = 0, limit: int = PAGE) -> dict:
        with self.lock:
            rows = self.db.execute(
                "SELECT seq, member, body FROM episodes WHERE seq > ? ORDER BY seq LIMIT ?",
                (after, limit + 1)).fetchall()
        more = len(rows) > limit
        rows = rows[:limit]
        episodes = [{**json.loads(body), "member": member} for _, member, body in rows]
        return {"episodes": episodes, "next": rows[-1][0] if rows else after, "more": more}

    def stats(self) -> dict:
        with self.lock:
            total = self.db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            per_member = dict(self.db.execute("SELECT member, COUNT(*) FROM episodes GROUP BY member").fetchall())
        return {"episodes": total, "members": len(self.members()), "contributors": len(per_member),
                "by_member": per_member}


def make_hub_handler(hub: Hub):
    class Handler(BaseHTTPRequestHandler):
        server_version = "pde-agent-hub/" + __version__

        def log_message(self, format, *args):
            pass

        def _json(self, status: HTTPStatus, data: dict) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _member(self) -> str | None:
            auth = self.headers.get("Authorization", "")
            name = hub.member_for(auth[7:].strip()) if auth.startswith("Bearer ") else None
            if name is None:
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "访问令牌不对或已被移除"})
            return name

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/api/v1/health":
                self._json(HTTPStatus.OK, {"ok": True, "version": __version__})
                return
            if url.path not in ("/api/v1/episodes", "/api/v1/stats"):
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            member = self._member()
            if member is None:
                return
            if url.path == "/api/v1/stats":
                self._json(HTTPStatus.OK, {**hub.stats(), "you": member})
                return
            query = parse_qs(url.query)
            try:
                after = max(0, int(query.get("after", ["0"])[0]))
                limit = min(PAGE, max(1, int(query.get("limit", [str(PAGE)])[0])))
            except ValueError:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "after 和 limit 必须是整数"})
                return
            self._json(HTTPStatus.OK, hub.download(after, limit))

        def do_POST(self):
            if urlparse(self.path).path != "/api/v1/episodes":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            member = self._member()
            if member is None:
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if not 0 < length <= MAX_BODY:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": f"请求体必须在 1 到 {MAX_BODY} 字节之间"})
                return
            try:
                payload = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeDecodeError):
                self._json(HTTPStatus.BAD_REQUEST, {"error": "请求不是合法的 JSON"})
                return
            episodes = payload.get("episodes") if isinstance(payload, dict) else None
            if not isinstance(episodes, list) or len(episodes) > MAX_BATCH:
                self._json(HTTPStatus.BAD_REQUEST, {"error": f"episodes 必须是最多 {MAX_BATCH} 条的列表"})
                return
            self._json(HTTPStatus.OK, hub.upload(member, episodes))

    return Handler


def make_hub_server(hub: Hub, host: str = "127.0.0.1", port: int = 8800) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_hub_handler(hub))
