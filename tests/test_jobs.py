"""v2.2.0: web solves are jobs in a child process, with progress, a hard timeout and cancel."""

import json
import os
import threading
import time
import urllib.error
import urllib.request

import pytest

from pde_agent.agent import PDEAgent
from pde_agent.app import AppState, make_server
from pde_agent.curriculum import basic_curriculum

POISSON = next(lesson for lesson in basic_curriculum() if lesson.problem.dim == 1 and not lesson.problem.time_dependent)
# no exact solution and a very tight tolerance: refines for a long time
SLOW = {"kind": "heat", "domain": [[0, 1]], "t_final": 2.0, "coefficients": {"alpha": 0.01},
        "initial": "sin(pi*x) + 0.3*sin(5*pi*x)", "name": "slow"}


@pytest.fixture
def server(tmp_path):
    servers = []

    def start(**kwargs):
        state = AppState(PDEAgent(home=tmp_path, time_budget=600), "deepseek", **kwargs)
        server = make_server(state, port=0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append((server, state))
        return f"http://127.0.0.1:{server.server_address[1]}", state

    yield start
    for server, state in servers:
        job = state.jobs.current()
        if job:
            state.jobs.cancel(job.id)
        server.shutdown()
        server.server_close()


def call(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def wait(base, job_id, limit=120):
    end = time.time() + limit
    while time.time() < end:
        status, job = call(f"{base}/api/jobs/{job_id}")
        assert status == 200
        if job["state"] != "running":
            return job
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_solve_runs_in_its_own_process_and_reports_progress(server):
    base, state = server()
    assert state.isolate
    status, data = call(base + "/api/solve", {"problem": POISSON.problem.to_dict(), "tol": POISSON.tol, "async": True})
    assert status == 200 and data["job"]["state"] == "running"
    job = wait(base, data["job"]["id"])
    assert job["state"] == "done", job
    report = job["result"]["report"]
    assert report["success"] and report["status"] == "verified"
    stages = [e["stage"] for e in job["events"]]
    assert stages[0] == "start" and "features" in stages and "run" in stages and stages[-1] == "done"
    # the costs are split by stage
    seconds = report["costs"]["seconds"]
    assert seconds["total"] >= seconds["solver"] > 0 and "verification" in seconds and "features" in seconds
    assert report["cost_text"].startswith("用时")
    # the server re-read the experience the job's process wrote
    status, stats = call(base + "/api/stats")
    assert stats["episodes"] == 1


def test_the_plain_endpoint_still_answers_with_the_report(server):
    base, _ = server()
    status, report = call(base + "/api/solve", {"problem": POISSON.problem.to_dict(), "tol": POISSON.tol})
    assert status == 200 and report["success"]


def test_a_job_that_runs_too_long_is_killed(server):
    base, state = server()
    status, data = call(base + "/api/solve", {"problem": SLOW, "tol": 1e-9, "async": True, "timeout": 4})
    assert status == 200
    job_id = data["job"]["id"]
    # the server keeps answering while the job runs
    assert call(base + "/api/info")[0] == 200
    status, busy = call(base + "/api/solve", {"problem": POISSON.problem.to_dict(), "tol": 1e-3})
    assert status == 409 and "取消" in busy["error"]
    pid = state.jobs._procs[job_id].pid
    job = wait(base, job_id, limit=30)
    assert job["state"] == "timeout" and "4 秒" in job["error"]
    time.sleep(0.5)
    assert not alive(pid)
    # and the next job runs
    status, report = call(base + "/api/solve", {"problem": POISSON.problem.to_dict(), "tol": POISSON.tol})
    assert status == 200 and report["success"]


def test_a_job_can_be_cancelled(server):
    base, state = server()
    status, data = call(base + "/api/solve", {"problem": SLOW, "tol": 1e-9, "async": True})
    job_id = data["job"]["id"]
    status, current = call(base + "/api/jobs/current")
    assert current["job"]["id"] == job_id
    time.sleep(1.0)
    pid = state.jobs._procs[job_id].pid
    status, job = call(f"{base}/api/jobs/{job_id}/cancel", {})
    assert status == 200 and job["state"] == "cancelled"
    time.sleep(0.5)
    assert not alive(pid)
    assert call(base + "/api/jobs/current")[1]["job"] is None


def test_the_sync_endpoint_reports_a_timeout(server):
    base, _ = server(job_timeout=3)
    status, data = call(base + "/api/solve", {"problem": SLOW, "tol": 1e-9})
    assert status == 504 and "3 秒" in data["error"]


def test_unknown_jobs_and_bad_timeouts(server):
    base, _ = server()
    assert call(base + "/api/jobs/0123456789ab")[0] == 404
    assert call(base + "/api/jobs/0123456789ab/cancel", {})[0] == 404
    status, data = call(base + "/api/solve", {"problem": POISSON.problem.to_dict(), "timeout": "soon"})
    assert status == 400 and "时间上限" in data["error"]
