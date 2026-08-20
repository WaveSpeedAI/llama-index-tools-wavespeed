"""Unit tests for WaveSpeedToolSpec. No live API calls are ever made."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from llama_index.core.tools.tool_spec.base import BaseToolSpec

from llama_index.tools.wavespeed import WaveSpeedToolSpec
from llama_index.tools.wavespeed.base import (
    DEFAULT_IMAGE_MODEL,
    DEFAULT_TIMEOUT,
    DEFAULT_VIDEO_MODEL,
)


class FakeClient:
    """Stand-in for ``wavespeed.Client`` recording what the spec sends."""

    def __init__(
        self,
        run_result: dict[str, Any] | None = None,
        run_error: Exception | None = None,
        get_result: dict[str, Any] | None = None,
        get_error: Exception | None = None,
    ) -> None:
        self.run_result = run_result if run_result is not None else {"outputs": []}
        self.run_error = run_error
        self.get_result_payload = get_result or {}
        self.get_error = get_error
        self.calls: list[dict[str, Any]] = []
        self.get_calls: list[str] = []

    def run(self, model, input, *, timeout=None, poll_interval=1.0):
        self.calls.append(
            {
                "model": model,
                "input": input,
                "timeout": timeout,
                "poll_interval": poll_interval,
            }
        )
        if self.run_error is not None:
            raise self.run_error
        return self.run_result

    def _get_result(self, request_id, timeout=None):
        self.get_calls.append(request_id)
        if self.get_error is not None:
            raise self.get_error
        return self.get_result_payload


def make_spec(**kwargs: Any) -> tuple[WaveSpeedToolSpec, FakeClient]:
    client = FakeClient(**kwargs)
    return WaveSpeedToolSpec(client=client), client


# -- spec wiring -----------------------------------------------------------


def test_is_a_base_tool_spec() -> None:
    spec, _ = make_spec()
    assert isinstance(spec, BaseToolSpec)


def test_tool_list_exposes_every_spec_function() -> None:
    spec, _ = make_spec()
    names = {t.metadata.name for t in spec.to_tool_list()}
    assert names == {
        "generate_image",
        "generate_video",
        "run_model",
        "get_prediction",
    }


def test_tool_schemas_do_not_expose_dropped_params() -> None:
    """`size`/`seed` are silently dropped by the API's input whitelist."""
    spec, _ = make_spec()
    tools = {t.metadata.name: t for t in spec.to_tool_list()}
    image_params = tools["generate_image"].metadata.fn_schema.model_fields
    assert set(image_params) >= {"prompt", "resolution", "aspect_ratio"}
    assert "size" not in image_params
    assert "seed" not in image_params
    video_params = tools["generate_video"].metadata.fn_schema.model_fields
    assert set(video_params) >= {"prompt", "duration"}
    assert "size" not in video_params
    assert "seed" not in video_params


def test_tool_descriptions_are_non_empty() -> None:
    spec, _ = make_spec()
    for tool in spec.to_tool_list():
        assert tool.metadata.description.strip()


# -- client construction ---------------------------------------------------


def test_builds_sdk_client_with_attribution_and_no_retries(monkeypatch) -> None:
    fake_client_cls = MagicMock(return_value=MagicMock())
    import wavespeed

    monkeypatch.setattr(wavespeed, "Client", fake_client_cls)
    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)

    WaveSpeedToolSpec(api_key="sk-test")

    fake_client_cls.assert_called_once_with(
        api_key="sk-test",
        client_name="llama-index",
        max_retries=0,
    )


def test_api_key_falls_back_to_environment(monkeypatch) -> None:
    fake_client_cls = MagicMock(return_value=MagicMock())
    import wavespeed

    monkeypatch.setattr(wavespeed, "Client", fake_client_cls)
    monkeypatch.setenv("WAVESPEED_API_KEY", "sk-env")

    WaveSpeedToolSpec()

    assert fake_client_cls.call_args.kwargs["api_key"] == "sk-env"


# -- argument mapping ------------------------------------------------------


def test_generate_image_defaults_and_mapping() -> None:
    spec, client = make_spec(run_result={"outputs": ["https://cdn/x.png"]})
    out = spec.generate_image("a red panda", resolution="2k", aspect_ratio="16:9")

    assert out == "https://cdn/x.png"
    call = client.calls[0]
    assert call["model"] == DEFAULT_IMAGE_MODEL
    assert call["input"] == {
        "prompt": "a red panda",
        "resolution": "2k",
        "aspect_ratio": "16:9",
    }
    assert call["timeout"] == DEFAULT_TIMEOUT
    assert call["poll_interval"] == 2.0


def test_generate_image_omits_unset_optional_args() -> None:
    spec, client = make_spec(run_result={"outputs": ["https://cdn/x.png"]})
    spec.generate_image("a red panda")
    assert client.calls[0]["input"] == {"prompt": "a red panda"}


def test_generate_video_defaults_and_mapping() -> None:
    spec, client = make_spec(run_result={"outputs": ["https://cdn/x.mp4"]})
    out = spec.generate_video("a drone shot over a glacier", duration=5)

    assert out == "https://cdn/x.mp4"
    assert client.calls[0]["model"] == DEFAULT_VIDEO_MODEL
    assert client.calls[0]["input"] == {
        "prompt": "a drone shot over a glacier",
        "duration": 5,
    }


def test_model_overrides_are_honoured() -> None:
    client = FakeClient(run_result={"outputs": ["https://cdn/x.png"]})
    spec = WaveSpeedToolSpec(
        client=client,
        image_model="wavespeed-ai/z-image/turbo",
        video_model="custom/video",
        timeout=None,
        poll_interval=0.5,
    )
    spec.generate_image("x")
    spec.generate_video("y")
    assert client.calls[0]["model"] == "wavespeed-ai/z-image/turbo"
    assert client.calls[1]["model"] == "custom/video"
    assert client.calls[0]["timeout"] is None
    assert client.calls[0]["poll_interval"] == 0.5


def test_run_model_passes_arbitrary_input() -> None:
    spec, client = make_spec(run_result={"outputs": ["https://cdn/x.png"]})
    spec.run_model("wavespeed-ai/z-image/turbo", {"prompt": "p", "steps": 8})
    assert client.calls[0]["model"] == "wavespeed-ai/z-image/turbo"
    assert client.calls[0]["input"] == {"prompt": "p", "steps": 8}


def test_run_model_accepts_a_json_string() -> None:
    spec, client = make_spec(run_result={"outputs": ["https://cdn/x.png"]})
    spec.run_model("m", '{"prompt": "p"}')
    assert client.calls[0]["input"] == {"prompt": "p"}


@pytest.mark.parametrize("bad", ["not json", "[1, 2]"])
def test_run_model_rejects_bad_json_string(bad: str) -> None:
    spec, client = make_spec()
    out = spec.run_model("m", bad)
    assert out.startswith("Error:")
    assert client.calls == []


# -- output formatting -----------------------------------------------------


def test_dict_outputs_are_reduced_to_urls() -> None:
    spec, _ = make_spec(
        run_result={"outputs": [{"url": "https://cdn/a.png"}, "https://cdn/b.png"]}
    )
    assert spec.generate_image("x") == "https://cdn/a.png\nhttps://cdn/b.png"


def test_non_url_output_falls_back_to_json() -> None:
    spec, _ = make_spec(run_result={"outputs": [{"text": "hello"}]})
    assert spec.generate_image("x") == '{"text": "hello"}'


def test_empty_outputs_are_reported_as_an_error() -> None:
    spec, _ = make_spec(run_result={"outputs": []})
    out = spec.generate_image("x")
    assert out.startswith("Error:")
    assert "no outputs" in out


# -- error surfacing -------------------------------------------------------


def test_run_error_surfaces_prediction_id_and_platform_text() -> None:
    spec, _ = make_spec(
        run_error=RuntimeError(
            "Prediction failed (task_id: abc-123): content policy violation"
        )
    )
    out = spec.generate_image("x")
    assert out.startswith("Error:")
    assert "abc-123" in out
    assert "content policy violation" in out
    assert DEFAULT_IMAGE_MODEL in out


def test_timeout_error_is_surfaced_not_raised() -> None:
    spec, _ = make_spec(
        run_error=TimeoutError("Prediction timed out after 600 seconds (task_id: t-9)")
    )
    out = spec.generate_video("x")
    assert out.startswith("Error:")
    assert "t-9" in out


# -- get_prediction --------------------------------------------------------


def test_get_prediction_returns_outputs_when_completed() -> None:
    spec, client = make_spec(
        get_result={
            "data": {"status": "completed", "outputs": [{"url": "https://cdn/v.mp4"}]}
        }
    )
    assert spec.get_prediction("p-1") == "https://cdn/v.mp4"
    assert client.get_calls == ["p-1"]


@pytest.mark.parametrize("status", ["failed", "cancelled", "timeout"])
def test_get_prediction_surfaces_terminal_failures(status: str) -> None:
    spec, _ = make_spec(
        get_result={"data": {"status": status, "error": "gpu exploded"}}
    )
    out = spec.get_prediction("p-2")
    assert out.startswith("Error:")
    assert status in out
    assert "gpu exploded" in out
    assert "p-2" in out


def test_get_prediction_terminal_failure_without_error_text() -> None:
    spec, _ = make_spec(get_result={"data": {"status": "failed"}})
    assert "Unknown error" in spec.get_prediction("p-3")


@pytest.mark.parametrize("status", ["created", "processing"])
def test_get_prediction_reports_in_flight_status(status: str) -> None:
    spec, _ = make_spec(get_result={"data": {"status": status}})
    out = spec.get_prediction("p-4")
    assert not out.startswith("Error:")
    assert status in out


def test_get_prediction_handles_completed_with_no_outputs() -> None:
    spec, _ = make_spec(get_result={"data": {"status": "completed", "outputs": []}})
    assert "no outputs" in spec.get_prediction("p-5")


def test_get_prediction_surfaces_transport_errors() -> None:
    spec, _ = make_spec(get_error=RuntimeError("HTTP 404: not found"))
    out = spec.get_prediction("p-6")
    assert out.startswith("Error:")
    assert "p-6" in out
    assert "404" in out


def test_get_prediction_prefers_a_public_sdk_getter() -> None:
    """If a future SDK adds a public get_result, use it over the private one."""
    client = FakeClient(get_result={"data": {"status": "completed", "outputs": ["u"]}})
    client.get_result = MagicMock(
        return_value={"data": {"status": "completed", "outputs": ["public"]}}
    )
    spec = WaveSpeedToolSpec(client=client)
    assert spec.get_prediction("p-7") == "public"
    assert client.get_calls == []
