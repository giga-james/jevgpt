# Terminal demo

The repository demo records a real interactive JevGPT session with [VHS](https://github.com/charmbracelet/vhs), then uses FFmpeg to produce a compact GIF and an H.264 MP4. The workflow follows [Making a Polished TUI Demo Video Without a Video Editor](https://blog.kunchenguid.com/p/making-a-polished-tui-demo-video).

The prompts are scripted; Jev's responses are live, not mocked or rewritten. Playback stays at real speed. The 24-token cap bounds each reply. VHS waits for the next `You:` prompt before typing another message, so differing response times do not interleave inputs.

## Record again

Install VHS, ttyd, FFmpeg, and uv. On macOS, `brew install vhs` installs the recording dependencies. This demo was captured with VHS 0.11.0; version 0.12.0 has an export cancellation bug. If needed, download [VHS 0.11.0](https://github.com/charmbracelet/vhs/releases/tag/v0.11.0) and run `make demo VHS=/path/to/vhs`. Then, from the repository root:

```sh
uv sync
# Populate the existing ignored .env with TYPESAFE_API_KEY.
make demo
```

Recording makes paid API requests using the configured key. It does not display the key or .env contents. The terminal contains only the launch command and two scripted chat turns. Outputs and timings vary between runs.

The resulting files are:

- `assets/jevgpt-demo.gif`: inline README animation.
- `assets/jevgpt-demo.mp4`: higher-quality video.
- `.cache/demo/raw.mp4`: original capture, ignored by Git.
- `.cache/demo/session.txt`: terminal text capture, ignored by Git.
- `.cache/demo/final.png`: final chat frame for visual review, ignored by Git.

To adjust encoding without making more API calls, run `make demo-render`. The capture uses a large font, a dark terminal, and a pink frame matching the README. The render only scales the recording and optimizes encoding; it does not change the displayed conversation or accelerate responses.
