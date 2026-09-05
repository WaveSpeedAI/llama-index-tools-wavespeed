"""WaveSpeed AI tool spec."""

import json
import os
from typing import Any

from llama_index.core.tools.tool_spec.base import BaseToolSpec

#: Channel-attribution name sent to the platform as ``X-Client-Name``.
_CLIENT_NAME = "llama-index"

#: Default wait deadline. ``None`` would block the calling agent forever, so a
#: finite default is used even though video jobs are slow.
DEFAULT_TIMEOUT = 600.0

#: Default text-to-image model.
DEFAULT_IMAGE_MODEL = "bytedance/seedream-v5.0-pro"

#: Default text-to-video model.
DEFAULT_VIDEO_MODEL = "wavespeed-ai/minimax-h3/text-to-video"


class WaveSpeedToolSpec(BaseToolSpec):
    """WaveSpeed AI tool spec.

    Gives an agent access to the WaveSpeed AI inference platform: text-to-image
    generation, text-to-video generation, running any model on the platform by
    id, and recovering the result of an earlier run from its prediction id.

    Setup:
        .. code-block:: bash

            pip install llama-index-tools-wavespeed
            export WAVESPEED_API_KEY="your-api-key"

    Usage:
        .. code-block:: python

            from llama_index.tools.wavespeed import WaveSpeedToolSpec

            tool_spec = WaveSpeedToolSpec()
            tools = tool_spec.to_tool_list()
    """

    spec_functions: list[str] = [
        "generate_image",
        "generate_video",
        "run_model",
        "get_prediction",
    ]

    def __init__(
        self,
        api_key: str | None = None,
        image_model: str = DEFAULT_IMAGE_MODEL,
        video_model: str = DEFAULT_VIDEO_MODEL,
        timeout: float | None = DEFAULT_TIMEOUT,
        poll_interval: float = 2.0,
        client: Any | None = None,
    ) -> None:
        """Initialize the WaveSpeed tool spec.

        Args:
            api_key: WaveSpeed API key. Falls back to the ``WAVESPEED_API_KEY``
                environment variable.
            image_model: Model id used by ``generate_image``.
            video_model: Model id used by ``generate_video``.
            timeout: Maximum seconds to wait for a prediction. Pass ``None`` to
                wait indefinitely; the task keeps running server-side either way.
            poll_interval: Seconds between result polls.
            client: Preconfigured ``wavespeed.Client`` (mainly for testing).
        """
        self.image_model = image_model
        self.video_model = video_model
        self.timeout = timeout
        self.poll_interval = poll_interval

        if client is None:
            from wavespeed import Client

            client = Client(
                api_key=api_key or os.environ.get("WAVESPEED_API_KEY"),
                client_name=_CLIENT_NAME,
                # Never let a host-configured default turn one tool call into a
                # second, separately billed submission.
                max_retries=0,
            )
        self.client = client

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _format_output(output: Any) -> str:
        """Render one platform output as a line an LLM can use."""
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            url = output.get("url")
            if isinstance(url, str):
                return url
        return json.dumps(output, default=str)

    @classmethod
    def _format_outputs(cls, outputs: list[Any]) -> str:
        return "\n".join(cls._format_output(o) for o in outputs)

    def _run(self, model: str, input: dict[str, Any]) -> str:
        payload = {k: v for k, v in input.items() if v is not None}
        try:
            result = self.client.run(
                model,
                payload,
                timeout=self.timeout,
                poll_interval=self.poll_interval,
            )
        except Exception as e:
            # The SDK's messages already carry "(task_id: ...)" and the
            # platform's error text; keep them verbatim so a failed paid task
            # stays traceable and the agent can call get_prediction later.
            return f"Error: WaveSpeed model {model!r} failed: {e}"
        outputs = result.get("outputs") or []
        if not outputs:
            return f"Error: WaveSpeed model {model!r} returned no outputs."
        return self._format_outputs(outputs)

    # -- tools -------------------------------------------------------------

    def generate_image(
        self,
        prompt: str,
        resolution: str | None = None,
        aspect_ratio: str | None = None,
    ) -> str:
        """
        Generate an image from a text prompt using WaveSpeed AI.

        Use this whenever the user asks for a picture, illustration, logo,
        artwork or any other still image. Returns the URL(s) of the generated
        image(s), one per line. Pass the URL back to the user exactly as
        returned; never modify or strip its query parameters.

        Args:
            prompt (str): Description of the image to generate. Be specific
                about subject, style, lighting and composition.
            resolution (str | None): Output resolution tier: "1k", "1.5k" or
                "2k". Higher tiers cost more. Defaults to the model's own default.
            aspect_ratio (str | None): Aspect ratio of the generated image,
                e.g. "1:1", "16:9", "9:16", "4:3". Defaults to the model's own
                default.

        """
        return self._run(
            self.image_model,
            {
                "prompt": prompt,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
            },
        )

    def generate_video(self, prompt: str, duration: int | None = None) -> str:
        """
        Generate a video from a text prompt using WaveSpeed AI.

        Use this whenever the user asks for a video, animation or motion clip.
        Video jobs take minutes, so expect this call to be slow. Returns the
        URL(s) of the generated video(s), one per line. Pass the URL back to the
        user exactly as returned.

        Args:
            prompt (str): Description of the video to generate, including the
                camera movement and action you want.
            duration (int | None): Video length in seconds (model-dependent,
                commonly 5 or 10). Defaults to the model's own default.

        """
        return self._run(self.video_model, {"prompt": prompt, "duration": duration})

    def run_model(self, model: str, input: dict[str, Any]) -> str:
        """
        Run any model on the WaveSpeed AI platform by its id.

        Use this only when the requested model is not covered by
        generate_image or generate_video - for example image editing,
        upscaling, image-to-video, speech or a specific named model. Browse
        the catalog and each model's input schema at https://wavespeed.ai/models.
        Returns the output URL(s), one per line.

        Args:
            model (str): WaveSpeed model id, e.g. "wavespeed-ai/z-image/turbo".
            input (dict[str, Any]): Input parameters for that model as a JSON
                object, e.g. {"prompt": "a lighthouse at dusk"}. Unknown keys
                are ignored by the platform.

        """
        if isinstance(input, str):
            # Many LLMs emit a JSON *string* where an object is expected.
            try:
                parsed = json.loads(input)
            except json.JSONDecodeError as e:
                return f"Error: 'input' must be a JSON object or valid JSON string: {e}"
            if not isinstance(parsed, dict):
                return (
                    "Error: 'input' must decode to a JSON object, "
                    f"got {type(parsed).__name__}."
                )
            input = parsed
        return self._run(model, input)

    def get_prediction(self, prediction_id: str) -> str:
        """
        Look up an earlier WaveSpeed prediction by its id.

        Use this to recover a run that timed out or whose result you lost. This
        matters most for video jobs, which often outlive a single tool call: if
        generate_video returned an error containing a task_id, pass that id here
        to check whether the job finished. Returns the output URL(s) one per
        line when complete, or a status line such as "created", "processing",
            "failed", "cancelled", "timeout" or "deleted".

        Args:
            prediction_id (str): The prediction / task id returned by a previous
                run, e.g. the value shown as "task_id: ..." in an error message.

        """
        try:
            result = self.client.get_result(prediction_id)
        except Exception as e:
            return f"Error: could not fetch WaveSpeed prediction {prediction_id!r}: {e}"

        data = result.get("data") or {}
        status = data.get("status") or "unknown"
        if status == "completed":
            outputs = data.get("outputs") or []
            if not outputs:
                return f"Prediction {prediction_id} completed but returned no outputs."
            return self._format_outputs(outputs)
        if status in ("failed", "cancelled", "timeout", "deleted"):
            error = data.get("error") or "Unknown error"
            return f"Error: prediction {prediction_id} {status}: {error}"
        return (
            f"Prediction {prediction_id} is not finished yet (status: {status}). "
            "Wait a little and check again."
        )
