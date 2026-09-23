# Four-token drafts with vocabulary expansion

Run `uv run jevgpt`: this is the default. Both drafting and verification use Jev. Chat output remains clean and only verified text is emitted. `--selection shortlist` restores the original heuristic-only baseline.

## One block

1. Draft up to **four tokens** sequentially with the original 254-token heuristic shortlist plus DONE. Stop drafting early on DONE or when the remaining output budget is exhausted. Keep the top eight original options at each position.
2. Retrieve **128 additional candidates per position**, excluding that position's entire original shortlist. Up to 96 come from lexical matches to the draft token, prompt, and preceding text: shared three-letter prefixes, exact case-insensitive word forms, and corpus-frequency tie-breaking. Fill remaining slots by seeded sampling from the wider eligible vocabulary. Retrieval runs locally; it is not another model call or a semantic embedding search.
3. Batch-rank those novel candidates with Jev. Every question includes only its own preceding answer prefix, not its proposed next token or later draft text. Questions are evaluated independently against shared prompt/history context. Preserve the best eight novel options at each position.
4. In a second batch, compare the original top eight and novel top eight together, plus DONE: at most 17 options per question. This re-evaluates competing text directly rather than comparing probabilities across candidate sets.
5. Emit the longest prefix where verifier and draft agree. At the first disagreement, emit the verifier's correction (or stop if it chooses DONE), discard every later speculative decision, and draft again from the corrected prefix. DONE proposed by the draft is also verified and may be overridden.

Each position has 382 candidate tokens evaluated across drafting and expansion, but only their shortlisted finalists participate in the final comparison. Expansion, not speculation itself, broadens vocabulary coverage. It remains a heuristic subset of the 99,412 eligible tokens, not an exhaustive search. Special tokens and partial UTF-8 fragments are still excluded.

## Latency and limitations

A full block normally takes four draft calls, one expansion batch, and one final verification batch: six sequential round trips for up to four accepted tokens. Request-size splitting or retries can add calls. Rejections waste later draft work; no speedup over plain shortlist is promised. Output arrives in verified chunks rather than streaming unverified draft tokens.

This is an application-level greedy draft-and-verify experiment. It does not implement distribution-preserving speculative sampling: candidate sets change, classifier probabilities are local to their questions, and there is no rejection-sampling correction. Both stages use the same model, so verification does not guarantee truthfulness or superior reasoning. Lexical alternatives can be irrelevant and random exploration can add noise.

The Python `Speculative` object records `drafted`, `accepted`, and `blocks` for measurements. Drafted and accepted counts include DONE decisions; suffix positions discarded after a disagreement count as drafted but not accepted. The CLI uses four draft positions, with fewer only at DONE or the output cap.

## Initial bounded comparison

Prompt: `Why is the sky blue? Answer in one short sentence.` Output cap: 12 tokens.

| Mode | Output | Calls | Reported input tokens | Time |
| --- | --- | ---: | ---: | ---: |
| Original shortlist | ` It is is appears appears is the` | 8 | 50,419 | 1.69 s |
| Four-token draft + expansion + verification | ` The sky is blue.` | 23 | 152,219 | 4.48 s |

Both selected DONE. Speculation accepted only 2 of 15 drafted decisions in this run. Its result was more grammatical but did not explain why the sky is blue. This is one observation, not a quality benchmark. It was slower than plain shortlist, while avoiding the tournament's exhaustive full-vocabulary workload.
