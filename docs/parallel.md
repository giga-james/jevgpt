# Four parallel stratified samples

Run `uv run jevgpt`. This is the default, with clean chat output. `--selection parallel` selects it explicitly; the old `--selection speculative` spelling is a compatibility alias for the same corrected algorithm.

## One output position

1. Build four **disjoint samples of 254 tokens**, covering 1,016 unique eligible vocabulary tokens.
2. Send four independent Jev requests concurrently. Every request sees the **same** user prompt, conversation history, and complete reply-so-far. Each chooses one best continuation from its sample.
3. Wait for all four winners, then make **one verifier request** with those four literal text fragments and DONE.
4. Append only the verifier's selected token, or stop on DONE. Recompute the samples and repeat at the next position.

This is five HTTP calls with two sequential stages, excluding retries. The HTTP pool and executor allow four requests in flight by default. `--concurrency 1` intentionally serializes the draft calls. Missing or invalid draft results fail the step rather than silently using fewer candidates. The four calls are separate requests, not a single batch pretending to be four calls.

The local probability of a winner is never compared against probabilities in another sample. The verifier directly compares all four fragments again. The four draft candidates are alternatives for **one** position, not four consecutive tokens. There is no draft acceptance/rejection loop or future-token suffix.

## Stratification

Candidates are drawn from the existing heuristic sources and distributed round-robin across the four samples. Per-sample quotas are:

| Source | Quota |
| --- | ---: |
| Tokens from the current prompt and recent answer | 32 |
| Corpus-frequency tokens, then basic characters/punctuation | 64 |
| Corpus n-gram continuations of the answer suffix | 32 |
| Lexical matches plus exploration from the wider vocabulary | 62 |
| Uniform exploration | 64 |

Tokens are globally deduplicated across all four samples. When a stratum lacks enough distinct candidates, remaining slots are filled with uniformly sampled unused vocabulary tokens. Each bucket is shuffled to reduce fixed position bias. `--seed` controls local sampling; it does not make Jev deterministic. `--corpus` changes the frequency and n-gram pools.

Lexical retrieval matches case-insensitive word forms and three-letter prefixes from the prompt and answer, with corpus-frequency tie-breaking. It does not use embeddings or another model. The underlying vocabulary still excludes special tokens, partial UTF-8 fragments, and nonprinting control characters; 99,412 tokens are eligible. Each step samples 1,016 of them rather than evaluating every token.

`--dry-run` reports sample sizes and unique coverage without API calls. `--trace` reveals local winners and the final choice; normal chat prints only the selected text. Output and repetition limits remain unchanged.

## Live check

Prompt: `Why is the sky blue? Answer in one short sentence.` Cap: 12 output tokens.

Output: `The sky is blue ForDue to the the the light of`

The run hit the 12-token cap, making 60 HTTP calls in 8.0 seconds (323,909 reported input tokens), about 0.67 seconds per emitted token. This is a single timing sample, not a latency guarantee. The output remains incomplete and repetitive; the test verifies the intended request structure, not a reliable quality improvement.
