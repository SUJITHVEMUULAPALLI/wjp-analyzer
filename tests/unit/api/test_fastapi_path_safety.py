import types

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient
from fastapi import HTTPException

from wjp_analyser.api import fastapi_app


def test_sanitize_filename_strips_path_and_adds_suffix(monkeypatch):
    fake_uuid = types.SimpleNamespace(hex="abc123")
    monkeypatch.setattr(fastapi_app.uuid, "uuid4", lambda: fake_uuid)

    safe = fastapi_app._sanitize_filename("../some/evil.DXF")
    assert safe == "evil_abc123.dxf"


def test_resolve_managed_path_allows_uploads_and_blocks_escape():
    managed_file = fastapi_app.UPLOAD_ROOT / "allowed.dxf"
    managed_file.write_text("content")

    resolved = fastapi_app._resolve_managed_path("allowed.dxf")
    assert resolved == managed_file.resolve()

    with pytest.raises(HTTPException):
        fastapi_app._resolve_managed_path("../../etc/passwd")

    managed_file.unlink()


def test_upload_endpoint_sanitises_filename(monkeypatch):
    fake_uuid = types.SimpleNamespace(hex="deadbeef")
    monkeypatch.setattr(fastapi_app.uuid, "uuid4", lambda: fake_uuid)

    client = TestClient(fastapi_app.app)
    response = client.post(
        "/upload",
        files={"file": ("../../blueprint.DXF", b"content")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "../../blueprint.DXF"
    assert payload["relative_path"] == "blueprint_deadbeef.dxf"
    saved_file = fastapi_app.UPLOAD_ROOT / payload["relative_path"]
    assert saved_file.exists()
    saved_file.unlink()
