<h1 align="center">JevGPT</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+">
  <a href="https://docs.typesafe.ai"><img src="https://img.shields.io/badge/Powered_by-Jev-f4a7c3" alt="Powered by Jev"></a>
  <img src="https://img.shields.io/badge/Status-Experimental-f4a7c3" alt="Experimental">
</p>

<p align="center"><strong>A classifier pretending to be a chatbot.</strong></p>

<p align="center">
  <img src="assets/jevgpt-banner.png" width="100%" alt="JevGPT: a smiling hamster surrounded by hand-drawn prompt, pick a token, append, repeat, and DONE cards.">
</p>

A deliberately questionable chatbot: Jev chooses one OpenAI tokenizer fragment at a time, and the growing answer becomes the next classification input. No generative model proposes the output.

## Run

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
# Only if you do not already have a .env file:
cp -n .env.template .env
# Add your TypeSafe key to .env, then:
uv run jevgpt "Say hello in one short sentence."
uv run jevgpt
```

Interactive mode supports `/reset` and `/quit`. Output streams as tokens arrive. By default, the CLI shows only the conversation. Add `--trace` to show routing diagnostics and a final summary of stopping reason, generated tokens, HTTP calls, reported input-token usage, and elapsed time. The key stays server-side in this local process; `.env` is ignored by Git.

```sh
uv run jevgpt "Why is the sky blue?" --max-tokens 64
uv run jevgpt "Why is the sky blue?" --selection shortlist
uv run jevgpt "Hello" --dry-run             # No Jev calls or API key needed
uv run jevgpt --corpus ./my-conversations.txt --seed 42
uv run pytest
```

The first run downloads the `cl100k_base` tokenizer data. `--dry-run` may therefore need network access on its first use. No OpenAI API key is needed.

## Algorithm

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#fff1f6", "primaryTextColor": "#29232b", "primaryBorderColor": "#d88bad", "lineColor": "#a7748c", "fontFamily": "sans-serif"}}}%%
flowchart LR
    P["Prompt + reply so far"] --> S["Stratified sampling<br/>4 disjoint sets × 254 tokens"]
    subgraph D["Speculative decoding experiment: parallel candidate drafts"]
        direction TB
        D1["Jev draft 1<br/>sample 1"]
        D2["Jev draft 2<br/>sample 2"]
        D3["Jev draft 3<br/>sample 3"]
        D4["Jev draft 4<br/>sample 4"]
    end
    S --> D1
    S --> D2
    S --> D3
    S --> D4
    D1 -- "candidate 1" --> V["Jev verifier<br/>4 candidates + DONE"]
    D2 -- "candidate 2" --> V
    D3 -- "candidate 3" --> V
    D4 -- "candidate 4" --> V
    V -- "One selected token" --> A["Append token to reply"]
    A -- "Next position" --> P
    V -- "DONE" --> E["Finish"]
    style E fill:#eaf5ef,stroke:#86ad95,color:#263a2e
```

**Speculative mode: four parallel drafts, one verifier, one output token.** All drafts target the same next-token position using identical prompt/reply context. This is a speculative-decoding-inspired candidate-selection experiment, rather than the paper’s exact decoding algorithm.

Four disjoint, stratified samples contain 254 candidates each. Four concurrent Jev calls choose one winner per sample for the same next-token position. A fifth call selects one of those four winners or DONE. Only that final token is appended, then the loop repeats.

See the [experiment log](docs/experiments.md) for the iterations, observed outputs, latency measurements, and why we changed direction.

### Selection modes

| Mode | How it works | Cost per step |
| --- | --- | --- |
| **Speculative (default)** | Four disjoint 254-token samples → four concurrent winners → one final choice plus `DONE`. | 5 calls in 2 sequential stages per token |
| **Tournament** | Evaluate all 390 buckets, compare their winners in semifinals, then choose a finalist or `DONE`. | Many batched calls; 44 per step in one live test |
| **Hierarchical** | Search vocabulary groups, keep three promising branches, then compare up to 48 tokens plus `DONE`. | Usually 4 HTTP calls |
| **Shortlist** | Choose from 254 tokens drawn from common text, the prompt, and likely continuations, plus `DONE`. | 1 HTTP call |

All modes use the prompt and full reply so far. Speculative mode makes four concurrent draft requests, then one verifier request. `--selection speculative` and `--selection parallel` select this same algorithm. Use `--selection shortlist` for the original one-call-per-token baseline. `--selection hierarchical` and `--selection tournament` remain available for experiments; tournament still launches all batches concurrently by default. Normal output stays a plain chatbot. `--trace` enables diagnostics. Retries and payload splitting can add calls.

Related reading: [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (Leviathan, Kalman, and Matias, 2023). Our Speculative mode uses four competing candidates for one position; it does not implement the paper’s distribution-preserving speculative decoding algorithm.

### Tradeoffs

- **Candidates, not future tokens:** the four drafts compete for the same next-token position. There is no sequential draft, acceptance loop, or discarded future suffix. Both draft and verifier use Jev; this is sampled tournament selection, not speculative decoding.
- **Coverage vs. speed:** tournament evaluates every eligible token at every step and is expensive. Hierarchy prunes branches, and shortlist is fastest but restricts candidates.
- **Search isn't certainty:** all approaches can discard a good continuation. Reported probabilities apply only to the choices shown, not the whole vocabulary.
- **Text has limits:** partial UTF-8 tokens are excluded, reducing multilingual coverage. Long conversations drop older turns; the current reply is retained.
- **Quality is experimental:** wider search hasn't demonstrated reliable improvement. One sky-blue answer was `Because the Air air air blocks blue`—still wrong and repetitive.

Generation stops on `DONE`, the output cap (128 tokens by default), or repeated loops. Local request-size guards keep payloads bounded but aren't exact Jev token counts.

[Speculative mode and benchmark](docs/parallel.md) · [Tournament details and benchmark](docs/tournament.md) · [Other algorithms and live comparisons](docs/algorithm.md) · [TypeSafe API](https://docs.typesafe.ai/api) · [Model limits](https://docs.typesafe.ai/models)
