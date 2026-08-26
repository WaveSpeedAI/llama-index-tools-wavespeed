# LlamaIndex Tools Integration: WaveSpeed

[![PyPI](https://img.shields.io/pypi/v/llama-index-tools-wavespeed.svg)](https://pypi.org/project/llama-index-tools-wavespeed/)

`WaveSpeedToolSpec` gives a LlamaIndex agent access to the
[WaveSpeed AI](https://wavespeed.ai) inference platform: text-to-image and
text-to-video generation, running any model in the catalog by id, and recovering
a long-running job from its prediction id.

It is built on the official [`wavespeed`](https://pypi.org/project/wavespeed/)
Python SDK, so it inherits the SDK's submission semantics — a submission POST is
never retried, terminal statuses (`failed` / `cancelled` / `timeout` / `deleted`) are handled
explicitly, and every request carries channel attribution.

## Installation

```bash
pip install llama-index-tools-wavespeed
export WAVESPEED_API_KEY="your-api-key"   # https://wavespeed.ai
```

## Tools

| Tool | What it does |
| --- | --- |
| `generate_image` | Text to image. Default model `bytedance/seedream-v5.0-pro`. Accepts `resolution` and `aspect_ratio`. |
| `generate_video` | Text to video. Default model `bytedance/seedance-2.5/text-to-video`. Accepts `duration`. |
| `run_model` | Runs any model id with an arbitrary input dict — image editing, upscaling, image-to-video, speech, and so on. See <https://wavespeed.ai/models>. |
| `get_prediction` | Fetches an earlier prediction by id. Video jobs routinely outlive a single tool call; when one times out the error carries a `task_id` the agent can pass here. |

Each tool returns the output URL(s), one per line, or a string starting with
`Error:` that carries the prediction id and the platform's own error text.

## Usage

```python
import asyncio

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.wavespeed import WaveSpeedToolSpec

tool_spec = WaveSpeedToolSpec()  # reads WAVESPEED_API_KEY
# or: WaveSpeedToolSpec(api_key="...")

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

response = asyncio.run(
    agent.run("Generate a 16:9 image of a red panda drinking boba tea.")
)
print(response)
```

`ReActAgent` works the same way:

```python
from llama_index.core.agent.workflow import ReActAgent

agent = ReActAgent(tools=tool_spec.to_tool_list(), llm=OpenAI(model="gpt-4.1"))
```

You can also call the tools directly, without an agent:

```python
tool_spec = WaveSpeedToolSpec()
print(tool_spec.generate_image("a lighthouse at dusk", aspect_ratio="16:9"))
print(tool_spec.run_model("wavespeed-ai/z-image/turbo", {"prompt": "a fox"}))
```

## Configuration

```python
WaveSpeedToolSpec(
    api_key=None,  # else WAVESPEED_API_KEY
    image_model="bytedance/seedream-v5.0-pro",
    video_model="bytedance/seedance-2.5/text-to-video",
    timeout=600.0,  # None waits forever
    poll_interval=2.0,
)
```

The default 600s timeout is deliberate: an unbounded wait would strand the
calling agent. When a video job exceeds it, the task keeps running server-side —
take the `task_id` out of the error and call `get_prediction`.

The underlying SDK client is created with `max_retries=0` so a single tool call
can never turn into a second, separately billed submission.

### Why `size` and `seed` are not exposed

The platform's input whitelist silently drops parameters the default models do
not declare. Advertising them in an LLM-facing schema would invite the agent to
set values that are then ignored, so `generate_image` and `generate_video`
expose only parameters that actually take effect. If a specific model you pick
does support them, pass them through `run_model`.

## A note on discovery

[llamahub.ai](https://llamahub.ai) is generated from metadata in the
`run-llama/llama_index` monorepo, which
[no longer accepts new integration packages](https://github.com/run-llama/llama_index/blob/main/CONTRIBUTING.md).
This package is therefore **not** listed on LlamaHub. It is a normal PyPI
distribution that installs into the `llama_index.tools` namespace package and
works exactly like the official tool specs.

## Development

```bash
uv sync --all-groups   # or: pip install -e . pytest ruff
pytest tests
ruff check .
```

The test suite mocks the SDK client and never calls the live API.

## License

MIT

---

**[WaveSpeed AI](https://wavespeed.ai/)** — AI image & video generation platform.
Try it in the browser: **[Image generator](https://wavespeed.ai/image-generator)** · **[Video generator](https://wavespeed.ai/video-generator)**
