from __future__ import annotations

import pytest
import torch
from fastapi.testclient import TestClient

from orion_100m.api import create_app as create_100m_app
from orion_100m.sdk import Orion100M, _validate_calibration_metadata
from orion_300m.api import create_app as create_300m_app
from orion_300m.config import Orion300MConfig
from orion_300m.model.decoder import Orion300MDecoder
from orion_300m.sdk import Orion300M
from orion_core.errors import ArtifactIntegrityError, ModelWeightsNotFound
from tests.test_tokenizer import synthetic_tokenizer


def test_missing_trained_weights_are_honest(tmp_path) -> None:
    with pytest.raises(ModelWeightsNotFound):
        Orion100M.load(weights_dir=tmp_path)
    with pytest.raises(ModelWeightsNotFound):
        Orion300M.load(tmp_path)


def test_100m_release_requires_calibrated_confidence_and_open_set_thresholds() -> None:
    thresholds, temperatures = _validate_calibration_metadata(
        {
            "unknown_max_known_probability": 0.7,
            "course_min_similarity": 0.25,
            "tags": {"exam": 0.6},
        },
        {"subject": 1.1, "document_type": 1.0, "topic": 1.2, "confidence": 0.9},
        {"tags": ["exam"]},
    )
    assert thresholds["unknown_max_known_probability"] == 0.7
    assert temperatures["confidence"] == 0.9
    with pytest.raises(ArtifactIntegrityError, match="unknown detection"):
        _validate_calibration_metadata(
            {"course_min_similarity": 0.2, "tags": {"exam": 0.5}},
            {"subject": 1, "document_type": 1, "topic": 1, "confidence": 1},
            {"tags": ["exam"]},
        )


def test_health_endpoints_report_missing_local_artifacts() -> None:
    client_100m = TestClient(create_100m_app())
    client_300m = TestClient(create_300m_app())
    assert client_100m.get("/health").json()["status"] == "unavailable"
    assert client_300m.get("/health").json()["status"] == "unavailable"
    response = client_100m.post("/v1/classify", json={"document": "x"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "ORION-E2001"


def test_no_context_answer_returns_abstention_without_model_call() -> None:
    config = Orion300MConfig(
        vocab_size=512,
        num_layers=1,
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=24,
        pretraining_length=4,
        context_length=12,
        max_generation_tokens=4,
    )
    with torch.random.fork_rng():
        model = Orion300MDecoder(config)
    calls = []
    model.register_forward_hook(lambda *_: calls.append(True))
    tokenizer = synthetic_tokenizer()
    manifest = {"model_version": "0.1.0", "status": "preview", "tokenizer_sha256": tokenizer.sha256}
    sdk = Orion300M(model, tokenizer, device=torch.device("cpu"), manifest=manifest)
    result = sdk.answer("What is supported?", documents=[])
    assert result.status == "no_context"
    assert result.text == ""
    assert not calls


def test_api_generation_schema_allows_empty_context_for_explicit_abstention() -> None:
    client = TestClient(create_300m_app())
    response = client.post(
        "/v1/generate", json={"prompt": "question", "documents": [], "task": "answer"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "no_context"
    assert response.json()["text"] == ""


def test_decoder_sdk_streams_real_cpu_generation_events() -> None:
    config = Orion300MConfig(
        vocab_size=512,
        num_layers=1,
        hidden_size=8,
        num_heads=2,
        num_kv_heads=1,
        intermediate_size=12,
        pretraining_length=4,
        context_length=256,
        max_generation_tokens=2,
    )
    model = Orion300MDecoder(config).eval()
    calls = []
    model.register_forward_hook(lambda *_: calls.append(True))
    tokenizer = synthetic_tokenizer()
    sdk = Orion300M(
        model,
        tokenizer,
        device=torch.device("cpu"),
        manifest={"model_version": "test", "status": "synthetic-test-fixture"},
    )
    events = list(sdk.stream_generate("hi", max_new_tokens=2, temperature=0, seed=4))
    assert events[-1]["event"] == "done"
    assert events[-1]["status"] == "generated"
    assert calls
    response = TestClient(create_300m_app(sdk)).post(
        "/v1/generate/stream",
        json={"prompt": "hi", "task": "generate", "max_new_tokens": 2, "seed": 4},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"event": "done"' in response.text


def test_batch_generation_preserves_order_and_api_no_context_abstains() -> None:
    config = Orion300MConfig(
        vocab_size=512,
        num_layers=1,
        hidden_size=8,
        num_heads=2,
        num_kv_heads=1,
        intermediate_size=12,
        pretraining_length=4,
        context_length=256,
        max_generation_tokens=2,
    )
    model = Orion300MDecoder(config).eval()
    calls = []
    model.register_forward_hook(lambda *_: calls.append(True))
    sdk = Orion300M(
        model,
        synthetic_tokenizer(),
        device=torch.device("cpu"),
        manifest={"model_version": "test", "status": "synthetic-test-fixture"},
    )
    results = sdk.generate_batch(
        [
            {"prompt": "first", "task": "generate", "max_new_tokens": 2, "seed": 3},
            {"prompt": "second", "task": "answer", "documents": []},
        ]
    )
    assert [result.request_id for result in results] == [None, None]
    assert [result.task for result in results] == ["generate", "answer"]
    assert [result.status for result in results] == ["generated", "no_context"]
    assert calls

    client = TestClient(create_300m_app(sdk))
    api_response = client.post(
        "/v1/generate:batch",
        json={
            "requests": [
                {"prompt": "first", "task": "generate", "max_new_tokens": 2, "seed": 3},
                {"prompt": "second", "task": "answer", "documents": []},
            ]
        },
    )
    assert api_response.status_code == 200
    assert [item["status"] for item in api_response.json()["results"]] == [
        "generated",
        "no_context",
    ]

    no_context_client = TestClient(create_300m_app())
    response = no_context_client.post(
        "/v1/generate:batch",
        json={
            "requests": [
                {"prompt": "one", "task": "answer"},
                {"prompt": "two", "task": "summarize"},
            ]
        },
    )
    assert response.status_code == 200
    assert [item["status"] for item in response.json()["results"]] == [
        "no_context",
        "no_context",
    ]


def test_api_caps_body_and_does_not_echo_invalid_input(monkeypatch) -> None:
    monkeypatch.delenv("ORION_API_TOKEN", raising=False)
    client = TestClient(create_100m_app())
    oversized = client.post("/v1/classify", content=b"x" * 2_500_001)
    assert oversized.status_code == 413
    invalid_value = "secret-input-that-must-not-be-echoed"
    invalid = client.post("/v1/classify", json={"document": invalid_value, "unrecognized": "bad"})
    assert invalid.status_code == 422
    assert invalid_value not in invalid.text


def test_api_rejects_unknown_generation_task_as_validation_error() -> None:
    client = TestClient(create_300m_app())
    response = client.post("/v1/generate", json={"prompt": "x", "task": "unknown", "documents": []})
    assert response.status_code == 422
