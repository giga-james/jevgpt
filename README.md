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

## Methodology

Jev is the core: it selects every candidate and decides which token to append. Speculative drafting and the tokenizer support that process.

| Component | Role |
| --- | --- |
| **Jev** | TypeSafe AI's classifier powers all four drafts and the final verifier. |
| **Speculative decoding** | Expands the candidate vocabulary: four parallel 254-token samples feed one verifier. |
| **OpenAI tokenizer** | `tiktoken` with `cl100k_base` supplies the token vocabulary and exact text fragments. |

### Architecture

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#fff1f6", "primaryTextColor": "#29232b", "primaryBorderColor": "#d88bad", "lineColor": "#a7748c", "fontFamily": "sans-serif"}}}%%
flowchart LR
    C["💬 Prompt + reply"]
    subgraph D["4 parallel drafts · 254 candidates each"]
        A["Jev ①"]
        B["Jev ②"]
        E["Jev ③"]
        F["Jev ④"]
    end
    O["OpenAI tokenizer<br/>cl100k_base vocabulary"] --> D
    C --> A & B & E & F
    A & B & E & F --> V{"🐹 Jev verifier<br/>Pick one candidate"}
    V --> T["✍️ Append token"]
    T -- "Repeat" --> C
    V -- "DONE" --> X(["✓ Finish"])
    classDef draft fill:#f0eaff,stroke:#a58bc9,color:#322747
    classDef verify fill:#fff0d9,stroke:#d3a05b,color:#493519
    classDef done fill:#eaf5ef,stroke:#86ad95,color:#263a2e
    class A,B,E,F draft
    class V verify
    class X done
```

**Speculative mode: draft in parallel → verify → append.** Four disjoint, stratified samples cover 1,016 tokens. Each Jev draft proposes one candidate for the same next-token position; the verifier selects one of the four candidates or `DONE`.

The main motivation for speculative mode is to expand the candidate vocabulary at each step: from a single 254-token shortlist to 1,016 distinct candidates across four samples. Each parallel draft predicts a possible next token from its sample, and the verifier selects which prediction advances the reply. The four predictions are alternatives for the same next position, rather than a sequence of four consecutive tokens.

See the [experiment log](docs/experiments.md) for the iterations, observed outputs, latency measurements, and why we changed direction.

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

## Selection modes

| Mode | How it works | Cost per step |
| --- | --- | --- |
| **Speculative (default)** | Four disjoint 254-token samples → four concurrent winners → one final choice plus `DONE`. | 5 calls in 2 sequential stages per token |
| **Tournament** | Evaluate all 390 buckets, compare their winners in semifinals, then choose a finalist or `DONE`. | Many batched calls; 44 per step in one live test |
| **Hierarchical** | Search vocabulary groups, keep three promising branches, then compare up to 48 tokens plus `DONE`. | Usually 4 HTTP calls |
| **Shortlist** | Choose from 254 tokens drawn from common text, the prompt, and likely continuations, plus `DONE`. | 1 HTTP call |

All modes use the prompt and full reply so far. Speculative mode makes four concurrent draft requests, then one verifier request. `--selection speculative` and `--selection parallel` select this same algorithm. Use `--selection shortlist` for the original one-call-per-token baseline. `--selection hierarchical` and `--selection tournament` remain available for experiments; tournament still launches all batches concurrently by default. Normal output stays a plain chatbot. `--trace` enables diagnostics. Retries and payload splitting can add calls.

Related reading: [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (Leviathan, Kalman, and Matias, 2023). Our Speculative mode uses four competing candidates for one position; it does not implement the paper’s distribution-preserving speculative decoding algorithm.

### Tradeoffs

- **Next-token predictions:** all four drafts predict a continuation at the same position; the verifier selects one to append. Both draft and verifier use Jev. This candidate-selection procedure differs from the paper's sequential drafting and acceptance algorithm.
- **Coverage vs. speed:** tournament evaluates every eligible token at every step and is expensive. Hierarchy prunes branches, and shortlist is fastest but restricts candidates.
- **Search isn't certainty:** all approaches can discard a good continuation. Reported probabilities apply only to the choices shown, not the whole vocabulary.
- **Text has limits:** partial UTF-8 tokens are excluded, reducing multilingual coverage. Long conversations drop older turns; the current reply is retained.
- **Quality is experimental:** wider search hasn't demonstrated reliable improvement. One sky-blue answer was `Because the Air air air blocks blue`—still wrong and repetitive.

Generation stops on `DONE`, the output cap (128 tokens by default), or repeated loops. Local request-size guards keep payloads bounded but aren't exact Jev token counts.

[Speculative mode and benchmark](docs/parallel.md) · [Tournament details and benchmark](docs/tournament.md) · [Other algorithms and live comparisons](docs/algorithm.md) · [TypeSafe API](https://docs.typesafe.ai/api) · [Model limits](https://docs.typesafe.ai/models)
