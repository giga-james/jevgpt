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

Hierarchical selection with clean chat output is the default. Run `uv run jevgpt` with no options to use it. Pass `--trace` to enable debugging output or `--selection shortlist` to use the original algorithm below.

1. Load `cl100k_base`; enumerate valid token IDs and retain independently valid UTF-8 text fragments. Exclude special tokens and nonprinting control characters. Preserve spaces and exact bytes.
2. Build a 254-token shortlist: 97 basic ASCII/whitespace tokens, up to 31 additional corpus-frequency tokens, 64 prompt/recent-output tokens, 46 n-gram continuations, and 16 corpus-frequency-weighted exploration tokens. Deduplicate; fill spare slots with common tokens, then seeded random vocabulary tokens. Shuffle options to reduce fixed ordering bias. These are quotas, not guaranteed counts per group.
3. Send the current prompt, recent conversation, and full answer-so-far to Jev. Ask a Choice question with those fragments plus `DONE`.
4. Validate the returned probability distribution, append its highest-probability fragment, and repeat.
5. Stop on `DONE`, the output cap (128 tokens by default, maximum 512), or six repeats of a 1–8 token cycle.

N-gram suggestions use suffixes of up to three output token IDs, backing off to two and one. The tiny bundled corpus is original example dialogue, not a representative language dataset. A larger custom corpus should improve the shortlist; quality is an experiment, not a guarantee. The seed controls local candidate sampling, not Jev inference determinism.

## Hierarchical vocabulary selection

The default hierarchical mode makes every eligible vocabulary token reachable without a corpus shortlist. It sorts exact text fragments lexicographically, partitions them into at most 32 contiguous groups, and recursively partitions each group until leaves contain at most 128 tokens. Every group describes its inclusive text boundaries, shared prefix, token count, and six illustrative examples. Corpus frequency selects examples only; it never excludes tokens from the tree.

For each output token, Jev chooses a group, then a subgroup, then an exact token from the leaf. Each decision sees the same prompt and full answer-so-far. Only the final token is appended. DONE is available at the root, where stopping is evaluated before committing to a group. Each emitted token currently takes three sequential decisions; DONE takes one. Retries can add HTTP calls. The existing output and repetition caps still apply.

Optional `--trace` output prints depth, selected option, option count, and the local probability of each routing decision. The final token probability is conditional on the selected leaf; routing probabilities are not multiplied or presented as global next-token probabilities. `--dry-run` prints the root's group descriptions without making API calls.

This expands vocabulary coverage, not necessarily answer quality. Greedy routing can choose the wrong branch and cannot recover within that token. Text ranges can be difficult for Jev to interpret, and tokens containing partial UTF-8 sequences remain excluded. This does not compute an exhaustive global argmax or perform beam search.

In an initial bounded comparison on “Why is the sky blue? Answer in one short sentence.”, hierarchical mode produced ` It Its Is It It It blue` (22 calls, 66,454 reported input tokens), while shortlist mode produced ` It is is blue.`. These are single-run observations, not a quality benchmark. Hierarchical mode is the default for experimenting with the broader vocabulary; use `--selection shortlist` to compare with the original mode.

## Limits and design tradeoffs

Jev allows at most 255 options per Choice. Its documented context limits are 64k tokens for a request and 32k for state plus the largest question. This implementation caps prompts at 8,000 UTF-8 bytes, keeps up to 4,000 bytes of recent complete conversation turns, and rejects serialized requests over 28,000 bytes. These are conservative local guards, not an exact Jev token counter. Very large generated answers or JSON-heavy inputs can hit the guard before the output cap. Context truncation only drops old turns; the current reply is never silently truncated.

Probabilities describe the presented classification choices, not underlying language-model logits. The shortlist can omit the best continuation. Filtering out partial UTF-8 tokens also reduces multilingual coverage. Shortlist mode uses one sequential API call per output token plus any DONE call or bounded rate-limit retries; hierarchical mode uses several. Every step resends context and candidates. `--trace` shows the selected candidate probability, not a global next-token probability. Trace output includes generated text, so treat it as conversation data.

The model defaults to the pinned `jev-1.13.0`; override with `JEV_MODEL`. HTTP 429 and 529 responses retry at most twice. Network failures and other API errors stop the run without substituting another model.

References: [TypeSafe API](https://docs.typesafe.ai/api), [model limits](https://docs.typesafe.ai/models), [OpenAI tokenizer guide](https://developers.openai.com/cookbook/examples/how_to_count_tokens_with_tiktoken).

## Initial live smoke tests

With the bundled corpus and Jev 1.13.0, one run of “Say hello in one short sentence.” produced ` Hello` and selected DONE on the next call. A run of “Why is the sky blue? Answer in one short sentence.” produced ` It is is blue.` and selected DONE after five output tokens. These demonstrate the loop and also its poor initial quality; they are observations, not deterministic expected outputs. Jev's rounded probability values sometimes sum to 0.99, so validation checks candidate membership, finite values in [0, 1], and a positive total instead of exact normalization.
