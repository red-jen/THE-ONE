"""
Dummy FastAPI + test pattern — keep this rhyme in your head:

    **ACE**  →  **A**pp  →  **C**all  →  **E**xpect

  • **A**pp     — build a FastAPI() and register a route (or import the real one).
  • **C**all    — TestClient(app).get("/path") or .post(...).
  • **E**xpect  — assert status_code and body.

This file uses a *fake* tiny app inside the test file so you see endpoint + test
in one place. Your real tests use the same ACE steps on `src.interface.api.app`.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient


# --- A: App (the "endpoint" lives here) --------------------------------------

dummy_app = FastAPI()


@dummy_app.get("/motto")
def read_motto():
    """Silly but memorable: one URL, one JSON blob."""
    return {"motto": "small tests, calm ships"}


@dummy_app.post("/echo")
def echo_message(payload: dict):
    """POST body in → same keys echoed back (good practice for POST tests)."""
    return {"you_sent": payload}


# --- C + E: Call + Expect (the test) -----------------------------------------


def test_get_motto_ace_pattern():
    # A — already done above (dummy_app + route)

    # C — Call the endpoint like a browser would (no real network)
    client = TestClient(dummy_app)
    response = client.get("/motto")

    # E — Expect: HTTP 200 and the JSON we promised
    assert response.status_code == 200
    assert response.json() == {"motto": "small tests, calm ships"}


def test_post_echo_round_trip():
    client = TestClient(dummy_app)
    response = client.post("/echo", json={"hello": "world"})

    assert response.status_code == 200
    assert response.json() == {"you_sent": {"hello": "world"}}











app = FastAPI()

app.get("users")
def get_users(users: List[User]):
    return users 


def test_get_users():

    client 