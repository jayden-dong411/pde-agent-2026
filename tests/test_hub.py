"""The experience hub and syncing with it, over real HTTP on free ports."""

import json
import os
import stat
import threading
import urllib.error
import urllib.request

import pytest

from pde_agent.agent import PDEAgent
from pde_agent.app import AppState, make_server
from pde_agent.cli import main
from pde_agent.cloud import CloudError, HubClient, SyncState, check_url, sync
from pde_agent.curriculum import basic_curriculum
from pde_agent.hub import Hub, make_hub_server
from pde_agent.memory import ExperienceStore, clean_episode, episode_id


@pytest.fixture
def hub(tmp_path):
    hub = Hub(tmp_path / "hub")
    server = make_hub_server(hub, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    hub.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield hub
    server.shutdown()
    server.server_close()


def poisson_lesson():
    return next(l for l in basic_curriculum() if l.problem.dim == 1 and not l.problem.time_dependent)


def solved_agent(home, lessons=1):
    agent = PDEAgent(home=home)
    for lesson in basic_curriculum()[:lessons]:
        agent.solve(lesson.problem, lesson.tol, learn=False)
    return agent


def test_members_and_tokens(tmp_path):
    hub = Hub(tmp_path / "hub")
    token = hub.add_member("小明")
    assert token.startswith("pdeh_") and hub.member_for(token) == "小明" and hub.member_for("pdeh_wrong") is None
    stored = (tmp_path / "hub" / "members.json").read_text(encoding="utf-8")
    assert token not in stored  # only a hash is kept
    assert stat.S_IMODE(os.stat(tmp_path / "hub" / "members.json").st_mode) == 0o600
    with pytest.raises(ValueError):
        hub.add_member("has space")
    assert hub.remove_member("小明") and hub.member_for(token) is None and not hub.remove_member("小明")


def test_two_people_share_experience(hub, tmp_path):
    alice = solved_agent(tmp_path / "alice", lessons=3)
    bob = PDEAgent(home=tmp_path / "bob")
    a = HubClient(hub.url, hub.add_member("alice"))
    b = HubClient(hub.url, hub.add_member("bob"))

    first = sync(alice.memory, alice.home, a)
    assert first.pushed == len(alice.memory.episodes) == 3 and first.pulled == 0  # own episodes are not "shared"
    assert sync(alice.memory, alice.home, a).pushed == 0  # nothing new the second time

    got = sync(bob.memory, bob.home, b)
    assert got.pulled == 3 and len(bob.memory.shared) == 3 and not bob.memory.episodes
    signature = alice.memory.episodes[0]["signature"]
    assert bob.memory.stats(signature) and bob.memory.stats(signature).keys() == alice.memory.stats(signature).keys()

    # bob's agent now starts from alice's grid instead of the coarse default
    lesson = poisson_lesson()
    spec = next(s for s in bob.registry if s.name == alice.memory.episodes[0]["method"])
    assert bob.memory.suggest_n(signature, spec, lesson.tol, "exact") is not None

    # a restart reads the shared file again; the state file remembers progress
    again = PDEAgent(home=tmp_path / "bob")
    assert len(again.memory.shared) == 3
    assert SyncState(tmp_path / "bob").for_hub(hub.url)["after"] == 3
    assert sync(again.memory, again.home, b).pulled == 0
    assert hub.stats() == {"episodes": 3, "members": 2, "contributors": 1, "by_member": {"alice": 3}}


def test_bad_records_are_refused(hub):
    token = hub.add_member("m")
    good = {"signature": "heat/1d/dirichlet", "tol": 1e-3, "success": True, "method": "heat.ftcs", "n": 32,
            "error": 1e-4, "methods": [{"method": "heat.ftcs", "outcome": "success", "n": 32, "error": 1e-4, "work": 10}]}
    bad = [
        {**good, "signature": "../../etc"},
        {**good, "tol": 5},
        {**good, "methods": [{"method": "x; rm -rf /", "outcome": "success", "n": 1}]},
        {**good, "methods": [{"method": "heat.ftcs", "outcome": "hacked", "n": 1}]},
        {**good, "problem": {"source": "x" * 5000}},
        "not an object",
    ]
    result = HubClient(hub.url, token).upload([good, *bad, good])
    assert result["accepted"] == 1 and result["duplicates"] == 1
    assert [r["index"] for r in result["rejected"]] == [1, 2, 3, 4, 5, 6]
    assert "code" not in json.dumps(hub.download()["episodes"])


def test_hub_needs_a_token(hub):
    with pytest.raises(CloudError, match="访问令牌"):
        HubClient(hub.url, "pdeh_not-a-member").stats()
    with urllib.request.urlopen(hub.url + "/api/v1/health") as response:
        assert json.loads(response.read())["ok"]
    request = urllib.request.Request(hub.url + "/api/v1/episodes", data=b"{}", method="POST",
                                     headers={"Authorization": "Bearer " + hub.add_member("m")})
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(request)
    assert exc.value.code == 400


def test_unreachable_hub_and_bad_urls(tmp_path):
    with pytest.raises(CloudError, match="连不上"):
        HubClient("http://127.0.0.1:9", "pdeh_x").stats()
    for url in ("ftp://x", "example.com", "http://x/?a=1"):
        with pytest.raises(ValueError):
            check_url(url)
    assert check_url(" https://pde.example.com:8800/ ") == "https://pde.example.com:8800"


def test_old_records_get_stable_ids(tmp_path):
    old = {"signature": "poisson/1d/dirichlet", "tol": 1e-3, "success": True, "method": "poisson.fd2", "n": 16,
           "methods": [{"method": "poisson.fd2", "outcome": "success", "n": 16, "work": 3.0}]}
    (tmp_path / "experience.jsonl").write_text(json.dumps(old) + "\n", encoding="utf-8")
    store = ExperienceStore(tmp_path / "experience.jsonl", tmp_path / "shared.jsonl")
    assert episode_id(store.episodes[0]) == episode_id(old) == clean_episode(old)["id"]
    assert store.add_shared([clean_episode(old)]) == 0  # already ours
    store.add({"signature": "heat/1d/dirichlet", "tol": 1e-3, "methods": []})
    assert len(store.episodes[-1]["id"]) == 32


def test_web_page_connects_and_syncs(hub, tmp_path):
    agent = solved_agent(tmp_path / "web")
    state = AppState(agent, "deepseek")
    server = make_server(state, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    try:
        status, data = call("/api/cloud")
        assert status == 200 and not data["configured"]
        assert call("/api/cloud/sync", {})[0] == 400
        assert call("/api/cloud", {"url": "nonsense", "token": "pdeh_whatever"})[0] == 400

        status, data = call("/api/cloud", {"url": hub.url + "/", "token": "pdeh_wrong-token-123", "enabled": True})
        assert status == 502 and "访问令牌" in data["error"]

        token = hub.add_member("web")
        status, data = call("/api/cloud", {"url": hub.url, "token": token, "enabled": True})
        assert status == 200, data
        assert data["url"] == hub.url and data["token"].startswith("pde") and token not in json.dumps(data)
        assert data["last_sync"] and data["hub"]["episodes"] == 1 and data["error"] is None
        assert hub.stats()["by_member"] == {"web": 1}

        friend = solved_agent(tmp_path / "friend", lessons=2)
        sync(friend.memory, friend.home, HubClient(hub.url, hub.add_member("friend")))
        status, data = call("/api/cloud/sync", {})
        assert status == 200 and data["shared"] == 2
        status, stats = call("/api/stats")
        assert stats["shared"] == 2 and stats["episodes"] == 1

        status, data = call("/api/cloud", {"clear": True})
        assert status == 200 and not data["configured"] and data["shared"] == 2
    finally:
        server.shutdown()
        server.server_close()


def test_cli_hub_and_sync(hub, tmp_path, capsys):
    data = str(tmp_path / "cli-hub")
    assert main(["hub", "--data", data, "add-member", "甲"]) == 0
    token = capsys.readouterr().out.strip().splitlines()[-1]
    assert token.startswith("pdeh_")
    assert main(["hub", "--data", data, "members"]) == 0 and "甲" in capsys.readouterr().out

    home = str(tmp_path / "cli-home")
    solved_agent(home)
    assert main(["--home", home, "sync"]) == 1  # nothing configured yet
    capsys.readouterr()
    assert main(["--home", home, "sync", "--url", hub.url, "--token", "pdeh_wrong-token-123"]) == 2
    assert "访问令牌" in capsys.readouterr().err
    from pde_agent.settings import Settings
    assert Settings(home).cloud == {}  # a failed first sync is not remembered

    assert main(["--home", home, "sync", "--url", hub.url, "--token", hub.add_member("cli")]) == 0
    assert "上传 1 条" in capsys.readouterr().out
    assert main(["--home", home, "sync"]) == 0 and "上传 0 条" in capsys.readouterr().out
