"""Provider contract tests run with a local MockTransport; no paid API calls."""

import json
from types import SimpleNamespace

import httpx
import pytest

from watch_intents.models import Candidate, FatalProviderError, TransientProviderError
from watch_intents.prompts import generation_messages, judgement_messages
from watch_intents.provider import LLMProvider


@pytest.fixture
def catalog():
    return {"version": "v0.1", "rules": [], "labels": [{"id": 0, "name": "OPEN_SPORT"}]}


@pytest.fixture
def root(tmp_path, monkeypatch):
    names = (
        "DASHSCOPE_API_KEY", "SUUNTO_CHAT_MODEL_CN_NAME", "SUUNTO_EVAL_JUDGE_PROVIDER",
        "SUUNTO_EVAL_JUDGE_NAME", "SUUNTO_EVAL_JUDGE_API_KEY",
        "SUUNTO_EVAL_JUDGE_BASE_URL", "DASHSCOPE_BASE_URL",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "dashscope.env").write_text(
        "DASHSCOPE_API_KEY=secret-generation\nSUUNTO_EVAL_JUDGE_API_KEY=secret-judge\n"
        "SUUNTO_CHAT_MODEL_CN_NAME=custom-generator\nSUUNTO_EVAL_JUDGE_NAME=custom-judge\n"
    )
    return tmp_path


@pytest.fixture
def slots():
    return [
        SimpleNamespace(
            slot_id=f"s{i}", locale="en", labels=("OPEN_SPORT",), kind="single",
            scenario="direct", family_id="family1", seed_brief="before a morning workout",
        )
        for i in (1, 2)
    ]


def candidate(sample_id="s1", **overrides):
    return {
        "sample_id": sample_id, "text": "Open the sport selection screen.",
        "labels": ["OPEN_SPORT"], "locale": "en", **overrides,
    }


def judgement(sample_id="s1", **overrides):
    return {
        "sample_id": sample_id, "labels": ["OPEN_SPORT"], "language_ok": True,
        "naturalness_ok": True, "ambiguous": False, "reason_codes": [], **overrides,
    }


def completion(content, **extras):
    return {
        "choices": [{"message": {"content": content}}],
        **extras,
    }


def response(rows, envelope="samples", **extras):
    return httpx.Response(200, json=completion(json.dumps({envelope: rows}), **extras))


def test_generate_and_account_for_known_usage(root, catalog, slots):
    requests = []

    def handler(request):
        requests.append(request)
        return response(
            [candidate("s2"), candidate("s1")],
            usage={"prompt_tokens": 11, "completion_tokens": 21, "total_tokens": 32},
        )

    provider = LLMProvider(root, catalog, transport=httpx.MockTransport(handler))
    result = provider.generate(slots)
    provider.close()
    assert [item.sample_id for item in result.items] == ["s1", "s2"]
    assert result.invalid_ids == []
    assert result.usage == {"prompt_tokens": 11, "completion_tokens": 21, "total_tokens": 32}
    assert result.request_count == 1
    body = json.loads(requests[0].content)
    assert body["model"] == "custom-generator"
    assert requests[0].headers["Authorization"] == "Bearer secret-generation"
    assert "temperature" not in body


@pytest.mark.parametrize("payload", [
    "not json with PRIVATE TEXT", "```json\n{}\n```", "null", "[]",
    '{"samples":null}', '{"samples":[],"unexpected":"PRIVATE TEXT"}',
])
def test_malformed_envelope_rejects_without_leaking(root, catalog, slots, payload, capsys):
    provider = LLMProvider(root, catalog, transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json=completion(payload))
    ))
    result = provider.generate(slots)
    assert result.invalid_ids == ["s1", "s2"]
    assert result.items == []
    assert "PRIVATE TEXT" not in repr(result)
    assert capsys.readouterr().out == ""
    provider.close()


@pytest.mark.parametrize("rows, valid, invalid", [
    ([candidate()], ["s1"], ["s2"]),
    ([candidate(), candidate(), candidate("s2")], ["s2"], ["s1"]),
    ([candidate(), candidate("unknown")], [], ["s1", "s2"]),
    ([candidate(text=7), candidate("s2")], ["s2"], ["s1"]),
    ([candidate(extra="PRIVATE TEXT"), candidate("s2")], ["s2"], ["s1"]),
    ([candidate(labels="OPEN_SPORT"), candidate("s2")], ["s2"], ["s1"]),
    ([None, {"sample_id": ["s1"]}, candidate("s2")], ["s2"], ["s1"]),
])
def test_rows_must_be_strict_complete_and_uniquely_identified(
    root, catalog, slots, rows, valid, invalid,
):
    provider = LLMProvider(root, catalog, transport=httpx.MockTransport(lambda _: response(rows)))
    result = provider.generate(slots)
    assert [item.sample_id for item in result.items] == valid
    assert result.invalid_ids == invalid
    assert "PRIVATE TEXT" not in repr(result)
    provider.close()


def test_judge_is_blind_and_does_not_receive_targets(root, catalog):
    requests = []
    sample = Candidate(**candidate(labels=["SECRET_TARGET_LABEL"]))

    def handler(request):
        requests.append(request)
        return response([judgement()], "judgements")

    provider = LLMProvider(root, catalog, transport=httpx.MockTransport(handler))
    result = provider.judge([sample])
    assert result.items[0].labels == ["OPEN_SPORT"]
    payload = json.loads(requests[0].content)
    assert payload["model"] == "custom-judge"
    assert requests[0].headers["Authorization"] == "Bearer secret-judge"
    assert "SECRET_TARGET_LABEL" not in json.dumps(payload)
    rows = json.loads(payload["messages"][1]["content"])["samples"]
    assert set(rows[0]) == {"sample_id", "text", "locale"}
    assert "untrusted data" in payload["messages"][0]["content"]
    provider.close()


@pytest.mark.parametrize("bad_fields", [
    {"language_ok": "true"}, {"ambiguous": 0}, {"reason_codes": ["PRIVATE TEXT"]},
    {"reason": "PRIVATE TEXT"},
])
def test_judge_boolean_and_reason_codes_are_strict(root, catalog, bad_fields):
    provider = LLMProvider(root, catalog, transport=httpx.MockTransport(
        lambda _: response([judgement(**bad_fields)], "judgements")
    ))
    result = provider.judge([Candidate(**candidate())])
    assert result.invalid_ids == ["s1"]
    assert "PRIVATE TEXT" not in repr(result)
    provider.close()


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 302])
def test_permanent_errors_fail_fast_and_never_quote_body(root, catalog, slots, status):
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(status, text="secret-generation PRIVATE TEXT")

    provider = LLMProvider(root, catalog, transport=httpx.MockTransport(handler))
    with pytest.raises(FatalProviderError) as caught:
        provider.generate(slots)
    assert len(attempts) == 1
    assert caught.value.request_count == 1
    assert str(status) in str(caught.value)
    assert "secret-generation" not in str(caught.value)
    assert "PRIVATE TEXT" not in str(caught.value)
    provider.close()


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retry_is_bounded_and_accounts_for_attempts(root, catalog, slots, monkeypatch, status):
    sleeps = []
    monkeypatch.setattr("watch_intents.provider.time.sleep", sleeps.append)
    provider = LLMProvider(root, catalog, transport=httpx.MockTransport(
        lambda _: httpx.Response(status, headers={"Retry-After": "600"}, json={"error": "secret"})
    ))
    with pytest.raises(TransientProviderError) as caught:
        provider.generate(slots)
    assert caught.value.request_count == 4
    assert sleeps == [60.0, 60.0, 60.0]
    assert caught.value.usage == {}
    provider.close()


def test_transient_failure_then_success(root, catalog, slots, monkeypatch):
    replies = iter([
        httpx.Response(429, headers={"Retry-After": "0"}),
        response([candidate(), candidate("s2")], usage={"total_tokens": 20}),
    ])
    monkeypatch.setattr("watch_intents.provider.time.sleep", lambda _: None)
    provider = LLMProvider(root, catalog, transport=httpx.MockTransport(lambda _: next(replies)))
    result = provider.generate(slots)
    assert result.request_count == 2
    assert result.usage == {"total_tokens": 20}
    provider.close()


def test_transport_exception_sanitized_and_smoke_can_disable_retries(root, catalog, slots):
    def handler(request):
        raise httpx.ReadTimeout("secret-generation PRIVATE TEXT", request=request)

    provider = LLMProvider(root, catalog, transport=httpx.MockTransport(handler), max_retries=0)
    with pytest.raises(TransientProviderError) as caught:
        provider.generate(slots)
    assert caught.value.request_count == 1
    assert "PRIVATE TEXT" not in str(caught.value)
    assert caught.value.__suppress_context__
    provider.close()


def test_empty_batch_does_not_make_a_request(root, catalog):
    def handler(request):
        pytest.fail("empty batch must not make an API request")

    provider = LLMProvider(root, catalog, transport=httpx.MockTransport(handler))
    assert provider.generate([]).request_count == 0
    assert provider.judge([]).request_count == 0
    provider.close()


def test_file_priority_and_shell_fallback_with_sanitized_metadata(root, catalog, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "shell-secret")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.com/v1/")
    provider = LLMProvider(root, catalog)
    assert provider.generator_key == "secret-generation"
    assert provider.metadata()["generator_base_url"] == "https://example.com/v1"
    assert "secret" not in json.dumps(provider.metadata())
    provider.close()


@pytest.mark.parametrize("name,value", [
    ("SUUNTO_CHAT_MODEL_CN_NAME", ""),
    ("SUUNTO_EVAL_JUDGE_NAME", " "),
    ("SUUNTO_EVAL_JUDGE_PROVIDER", "other"),
    ("DASHSCOPE_API_KEY", ""),
    ("DASHSCOPE_BASE_URL", "https://example.com/v1?key=secret"),
    ("SUUNTO_EVAL_JUDGE_BASE_URL", "https://user:secret@example.com/v1"),
])
def test_bad_configuration_has_only_safe_errors(root, catalog, monkeypatch, name, value):
    monkeypatch.setenv(name, "stale-shell-value")
    path = root / "dashscope.env"
    path.write_text(path.read_text() + f"{name}={value}\n")
    with pytest.raises(FatalProviderError) as caught:
        LLMProvider(root, catalog)
    assert "secret" not in str(caught.value)
    assert value not in str(caught.value) or not value.strip()


@pytest.mark.parametrize("usage", [None, {}, {"total_tokens": None}, {"total_tokens": True}])
def test_unknown_usage_is_not_reported_as_zero(root, catalog, slots, usage):
    provider = LLMProvider(root, catalog, transport=httpx.MockTransport(
        lambda _: response([candidate(), candidate("s2")], usage=usage)
    ))
    assert provider.generate(slots).usage == {}
    provider.close()


def test_prompts_preserve_target_scenarios_but_only_judge_text(catalog, slots):
    generation = generation_messages(slots, catalog)
    data = json.loads(generation[1]["content"])["slots"]
    assert data[0]["sample_id"] == "s1"
    assert data[0]["target_labels"] == ["OPEN_SPORT"]
    assert data[0]["seed_brief"] == "before a morning workout"
    judge = judgement_messages([Candidate(**candidate())], catalog)
    assert "target_labels" not in judge[1]["content"]
    assert "Negation does not imply the opposite" in judge[0]["content"]
    assert "Resolve cancellations and corrections by scope" in judge[0]["content"]
    assert "and ambiguous false by contract" in judge[0]["content"]
    assert "do not assign unsupported_parameter merely" in judge[0]["content"]
    assert "within 512 XLM-R tokens" in generation[0]["content"]
    assert "character counts are not token counts" in generation[0]["content"]


@pytest.mark.parametrize("name", ["SUUNTO_CHAT_MODEL_CN_NAME", "SUUNTO_EVAL_JUDGE_NAME"])
def test_missing_models_have_no_hardcoded_defaults(root, catalog, name):
    path = root / "dashscope.env"
    path.write_text("\n".join(
        line for line in path.read_text().splitlines() if not line.startswith(name + "=")
    ))
    with pytest.raises(FatalProviderError, match=f"Missing model: {name}"):
        LLMProvider(root, catalog)


def test_changed_file_drives_both_requests_and_metadata(root, catalog, slots, monkeypatch):
    settings = {
        "DASHSCOPE_API_KEY": "new-generator-key",
        "SUUNTO_EVAL_JUDGE_API_KEY": "new-judge-key",
        "SUUNTO_CHAT_MODEL_CN_NAME": "new-generator",
        "SUUNTO_EVAL_JUDGE_NAME": "new-judge",
        "DASHSCOPE_BASE_URL": "https://generator.example/v1/",
        "SUUNTO_EVAL_JUDGE_BASE_URL": "https://judge.example/v1/",
        "SUUNTO_EVAL_JUDGE_PROVIDER": "dashscope",
    }
    for name in settings:
        monkeypatch.setenv(name, "stale-shell-value")
    (root / "dashscope.env").write_text(
        "\n".join(f"{name}={value}" for name, value in settings.items())
    )
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.host == "generator.example":
            return response([candidate(), candidate("s2")])
        return response([judgement()], "judgements")

    provider = LLMProvider(root, catalog, transport=httpx.MockTransport(handler))
    provider.generate(slots)
    provider.judge([Candidate(**candidate())])
    for request, role in zip(requests, ("generator", "judge"), strict=True):
        assert str(request.url) == f"https://{role}.example/v1/chat/completions"
        assert json.loads(request.content)["model"] == f"new-{role}"
        assert request.headers["Authorization"] == f"Bearer new-{role}-key"
        assert provider.metadata()[f"{role}_model"] == f"new-{role}"
    assert "-key" not in json.dumps(provider.metadata())
    provider.close()


@pytest.mark.parametrize("labels", [[0], [0, 1]])
def test_numeric_labels_are_rejected_with_safe_diagnostics(root, catalog, labels):
    provider = LLMProvider(root, catalog, transport=httpx.MockTransport(
        lambda _: response([candidate(labels=labels), candidate("s2", labels=[])])
    ))
    from types import SimpleNamespace
    slots = [SimpleNamespace(slot_id=f"s{i}", locale="en", labels=(), kind="zero",
                             scenario="direct", family_id="family", seed_brief="seed")
             for i in (1, 2)]
    result = provider.generate(slots)
    assert result.invalid_ids == ["s1"]
    assert result.invalid_reasons == {"s1": ["schema_labels_string_type"]}
    assert [row.sample_id for row in result.items] == ["s2"]
    provider.close()


def test_prompts_explicitly_require_label_names_and_json_mode(root, catalog, slots):
    requests = []
    def handler(request):
        requests.append(request)
        return response([candidate(), candidate("s2")])
    provider = LLMProvider(root, catalog, transport=httpx.MockTransport(handler))
    provider.generate(slots)
    body = json.loads(requests[0].content)
    assert body["response_format"] == {"type": "json_object"}
    assert "never numeric IDs" in body["messages"][0]["content"]
    assert "never numeric IDs" in judgement_messages([Candidate(**candidate())], catalog)[0]["content"]
    provider.close()


def test_truncated_valid_json_is_not_accepted(root, catalog, slots):
    payload = completion(json.dumps({"samples": [candidate(), candidate("s2")]}))
    payload["choices"][0]["finish_reason"] = "length"
    provider = LLMProvider(root, catalog, transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json=payload)
    ))
    result = provider.generate(slots)
    assert result.items == []
    assert all(codes == ["output_truncated"] for codes in result.invalid_reasons.values())
    provider.close()



def test_diagnostics_do_not_include_unknown_field_names_or_input_text(root, catalog, slots):
    provider = LLMProvider(root, catalog, transport=httpx.MockTransport(
        lambda _: response([candidate(PRIVATE_REJECTED_SENTINEL="private text")])
    ))
    result = provider.generate(slots)
    assert "PRIVATE_REJECTED_SENTINEL" not in repr(result)
    assert "private text" not in repr(result)
    assert result.invalid_reasons["s1"] == ["schema_unknown_field_extra_forbidden"]
    provider.close()



def test_http_429_lowers_only_affected_model_capacity(root, catalog, slots, monkeypatch):
    import threading

    from watch_intents.runtime import RuntimeSettings
    replies = iter([httpx.Response(429, headers={"Retry-After": "0"}), response([candidate(), candidate("s2")])])
    events = []
    provider = LLMProvider(root, catalog, transport=httpx.MockTransport(lambda _: next(replies)))
    monkeypatch.setattr("watch_intents.provider._delay", lambda *args: 0)
    provider.configure_parallel(RuntimeSettings(), threading.Event(), events.append)
    assert len(provider.generate(slots).items) == 2
    gate = provider._gates[(provider.generator_base_url, provider.generator_model)]
    other = provider._gates[(provider.judge_base_url, provider.judge_model)]
    assert gate.snapshot()["active_limit"] == 1
    assert other.snapshot()["active_limit"] == 2
    assert events[0]["event"] == "rate_limited"
    provider.close()


def test_identical_model_shares_gate_despite_different_credentials(root, catalog):
    import threading

    from watch_intents.runtime import RuntimeSettings
    provider = LLMProvider(root, catalog)
    provider.judge_model = provider.generator_model
    provider.configure_parallel(RuntimeSettings(generator_rpm=6, judge_rpm=4), threading.Event(), lambda e: None)
    assert len(provider._gates) == 1
    assert next(iter(provider._gates.values())).snapshot()["rpm"] == 4
    provider.close()



def test_too_small_tpm_fails_without_api_call_or_fake_request_count(root, catalog, slots):
    import threading

    from watch_intents.runtime import RuntimeSettings
    provider = LLMProvider(root, catalog, transport=httpx.MockTransport(
        lambda _: pytest.fail("budget failure must not reach API")
    ))
    provider.configure_parallel(RuntimeSettings(generator_tpm=1), threading.Event(), lambda e: None)
    with pytest.raises(FatalProviderError) as caught:
        provider.generate(slots)
    assert caught.value.request_count == 0
    provider.close()



def test_key_rotation_never_changes_provider_identity(root, catalog):
    first = LLMProvider(root, catalog)
    original = first.metadata()
    first.close()
    path = root / "dashscope.env"
    path.write_text(path.read_text().replace("secret-generation", "rotated-generation-key")
                    .replace("secret-judge", "rotated-judge-key"))
    second = LLMProvider(root, catalog)
    assert second.metadata() == original
    assert second.generator_key == "rotated-generation-key"
    second.close()
