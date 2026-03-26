import importlib
import os

from fastapi.testclient import TestClient


def load_api_module(auth_enabled: bool):
    if auth_enabled:
        os.environ["HB_API_AUTH"] = "true"
    else:
        os.environ["HB_API_AUTH"] = "false"

    os.environ["HB_API_USERNAME"] = "admin"
    os.environ["HB_API_PASSWORD"] = "admin"
    os.environ["HB_JWT_SECRET"] = "test-secret"

    import src.interface.api as api_mod

    return importlib.reload(api_mod)


def test_health_endpoint_available():
    api_mod = load_api_module(auth_enabled=False)
    client = TestClient(api_mod.app)

    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data


def test_auth_token_and_protected_route():
    api_mod = load_api_module(auth_enabled=True)
    client = TestClient(api_mod.app)

    token_resp = client.post("/auth/token", json={"username": "admin", "password": "admin"})
    assert token_resp.status_code == 200
    token = token_resp.json()["access_token"]

    no_auth_resp = client.post(
        "/run-multicam-path",
        json={"run_name": "x", "videos": [], "videos_dir": None},
    )
    assert no_auth_resp.status_code == 401

    auth_resp = client.post(
        "/run-multicam-path",
        headers={"Authorization": f"Bearer {token}"},
        json={"run_name": "x", "videos": [], "videos_dir": None},
    )
    assert auth_resp.status_code == 400
