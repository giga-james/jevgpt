# Algorithm details

The default is now the [four parallel stratified samples](parallel.md), with clean chat output. This page documents the alternative `--selection hierarchical` and `--selection shortlist` modes and their earlier benchmarks.

### Optional shortlist mode

1. Load `cl100k_base`; enumerate valid token IDs and retain independently valid UTF-8 text fragments. Exclude special tokens and nonprinting control characters. Preserve spaces and exact bytes.
2. Build a 254-token shortlist: 97 basic ASCII/whitespace tokens, up to 31 additional corpus-frequency tokens, 64 prompt/recent-output tokens, 46 n-gram continuations, and 16 corpus-frequency-weighted exploration tokens. Deduplicate; fill spare slots with common tokens, then seeded random vocabulary tokens. Shuffle options to reduce fixed ordering bias. These are quotas, not guaranteed counts per group.
3. Send the current prompt, recent conversation, and full answer-so-far to Jev. Ask a Choice question with those fragments plus `DONE`.
4. Validate the returned probability distribution, append its highest-probability fragment, and repeat.
5. Stop on `DONE`, the output cap (128 tokens by default, maximum 512), or six repeats of a 1–8 token cycle.

N-gram suggestions use suffixes of up to three output token IDs, backing off to two and one. The tiny bundled corpus is original example dialogue, not a representative language dataset. A larger custom corpus should improve the shortlist; quality is an experiment, not a guarantee. The seed controls local candidate sampling, not Jev inference determinism.

## Hierarchical vocabulary selection

The optional hierarchical mode makes every eligible vocabulary token reachable without a corpus shortlist. It sorts exact text fragments lexicographically, partitions them into at most 32 contiguous groups, and recursively partitions each group until leaves contain at most 128 tokens. Every group describes its inclusive text boundaries, shared prefix, token count, and six illustrative examples. Corpus frequency selects examples only; it never excludes tokens from the tree.

For each output token, a bounded beam search keeps up to three promising branches at every level. Child routing scores are normalized locally and multiplied by their parent's score to rank the expanded branches; these products are heuristics, not calibrated global probabilities. Once the beam reaches leaves, Jev ranks the actual fragments in each leaf and retains up to 16 per leaf. A final fresh Choice compares at most 48 explicit fragments together, plus DONE. Routing scores are not included in this final comparison.

Each decision sees the same prompt and full answer-so-far. Only the final selected token is appended. DONE is considered only in the final comparison, so it competes against actual continuations rather than broad vocabulary groups. With the current tree and defaults, each generated token or DONE normally takes four sequential HTTP calls: root ranking, batched subgroup rankings, batched leaf rankings, and final selection. Sibling questions share the same state in one request and are evaluated independently by Jev. The search still performs the same eight classification decisions; it no longer waits for a separate round trip for each sibling. Large batches are split at a conservative 56,000-byte request cap, and each state-plus-question is limited to 28,000 bytes. Payload splitting and retries can add calls. The existing output and repetition caps still apply. The Python constructor exposes `beam_width` (default 3, maximum 8) and `per_leaf` (default 16); their product cannot exceed 254.

Optional `--trace` output prints depth, retained options, option count, and local probabilities. The final token probability is conditional on the finalist set, not the entire vocabulary. `--selection hierarchical --dry-run` prints the first routing question's group descriptions without making API calls. Normal chat output stays clean.

This reduces early lock-in but cannot eliminate it: pruning can still discard the best branch, and ranking within a leaf can discard a good token. Multiple surviving paths can share a parent. Lexicographic groups remain awkward for Jev to interpret; tokens containing partial UTF-8 sequences remain excluded. This is beam search over vocabulary branches for a single next token, not beam search over alternative complete replies or an exhaustive global argmax.

The original greedy hierarchy is retained as the Python method `choose_greedy` for controlled comparisons; the CLI's optional hierarchical mode uses the new multi-branch selector. `--selection shortlist` still selects the original corpus shortlist.

Earlier bounded live comparison before HTTP batching (16-token cap, one run per prompt and mode):

| Prompt | Greedy hierarchy | Multi-branch hierarchy | HTTP calls (greedy / multi-branch) |
| --- | --- | --- | --- |
| Why is the sky blue? Answer in one short sentence. | ` It Its Is It It It It blue` | ` Because the Air air air blocks blue` | 25 / 64 |
| Say hello in one short sentence. | `Hello` | `Hello` | 4 / 16 |

All four runs selected DONE. The sky explanation remains factually incorrect and repetitive. These observations verify operation and show the call-cost tradeoff; they do not establish a reliable quality improvement.

After batching sibling questions, an eight-token sky-prompt comparison took 11.86 seconds / 64 HTTP calls with serial ranking and 6.55 seconds / 32 HTTP calls with batched ranking (about 45% less elapsed time). Both runs hit the eight-token cap and remained repetitive. This is a single live timing sample, not a latency guarantee; no vocabulary coverage or beam width was reduced.

## Limits and design tradeoffs

Jev allows at most 255 options per Choice. Its documented context limits are 64k tokens for a request and 32k for state plus the largest question. This implementation caps prompts at 8,000 UTF-8 bytes, keeps up to 4,000 bytes of recent complete conversation turns, and rejects individual state-plus-question payloads over 28,000 bytes (batched requests are capped at 56,000 bytes). These are conservative local guards, not an exact Jev token counter. Very large generated answers or JSON-heavy inputs can hit the guard before the output cap. Context truncation only drops old turns; the current reply is never silently truncated.

Probabilities describe the presented classification choices, not underlying language-model logits. The shortlist can omit the best continuation. Filtering out partial UTF-8 tokens also reduces multilingual coverage. Shortlist mode uses one sequential API call per output token plus any DONE call or bounded rate-limit retries; hierarchical mode uses several. Every step resends context and candidates. `--trace` shows the selected candidate probability, not a global next-token probability. Trace output includes generated text, so treat it as conversation data.

The model defaults to the pinned `jev-1.13.0`; override with `JEV_MODEL`. HTTP 429 and 529 responses retry at most twice. Network failures and other API errors stop the run without substituting another model.

References: [TypeSafe API](https://docs.typesafe.ai/api), [model limits](https://docs.typesafe.ai/models), [OpenAI tokenizer guide](https://developers.openai.com/cookbook/examples/how_to_count_tokens_with_tiktoken).

## Initial live smoke tests

With the bundled corpus and Jev 1.13.0, one run of “Say hello in one short sentence.” produced ` Hello` and selected DONE on the next call. A run of “Why is the sky blue? Answer in one short sentence.” produced ` It is is blue.` and selected DONE after five output tokens. These demonstrate the loop and also its poor initial quality; they are observations, not deterministic expected outputs. Jev's rounded probability values sometimes sum to 0.99, so validation checks candidate membership, finite values in [0, 1], and a positive total instead of exact normalization.
