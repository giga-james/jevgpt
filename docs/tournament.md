# Full-vocabulary tournament

Run `uv run jevgpt --selection tournament`. Tournament is optional; the CLI shows only chat text unless `--trace` is requested. `--selection tournament --dry-run` reports vocabulary coverage and bucket count without making Jev calls.

## Selection

1. Enumerate the existing `cl100k_base` vocabulary of independently valid UTF-8 text fragments. There are currently 99,412 eligible tokens. Special tokens, partial UTF-8 fragments, and nonprinting control characters remain excluded.
2. Shuffle deterministically with `--seed` and divide into disjoint buckets of at most 255 tokens. Each bucket is also capped at approximately 12,000 serialized bytes. The current vocabulary produces 390 buckets. Every eligible token appears exactly once in the first round; corpus frequency does not filter candidates.
3. Evaluate every bucket against the same current user prompt, recent chat, and full answer-so-far. Each question asks for the best literal continuation at the same next-token position. Advance the local argmax from each bucket, regardless of its probability relative to other buckets.
4. If more than 254 winners remain, partition and re-evaluate them. Currently 390 first-round winners enter two semifinals (255 and 135 options).
5. Compare the remaining finalists together with DONE. Append only this final winner, or stop on DONE. Repeat the entire tournament at the next position.

This is a tournament, not speculative decoding of multiple future positions. Probabilities from separate buckets are never compared against each other. The final probability is conditional on the finalists, not a global next-token probability. Grouping can affect outcomes, and a good candidate eliminated in an early round cannot recover. Full first-round coverage does not guarantee a global argmax or coherent answers.

## Execution and limits

Independent questions are packed into requests up to 56,000 serialized UTF-8 bytes, with a separate 28,000-byte state-plus-question guard. All independent tournament batches run concurrently by default, with a 390-request ceiling and a matching HTTP connection-pool limit. Multiple bucket questions still share each request, so 390 buckets usually require far fewer than 390 concurrent requests. Pass `--concurrency 4` to restore the earlier throttle. This submits all first-round work without a four-request queue, but does not guarantee simultaneous server execution. Jev may throttle bursts. Responses are mapped back by question ID even when requests finish out of order. Call and usage counters are protected across threads. On a batch failure, queued work is canceled and the generation fails rather than silently omitting vocabulary buckets; already running calls may finish.

These byte guards are conservative local limits, not Jev tokenizer counts. Rate-limit retries and growing context can increase requests and latency. Generation still uses the existing output and repetition caps. The custom corpus affects the older selection modes; tournament bucket membership depends only on eligible tokens and the shuffle seed.

The exhaustive first round is expensive even when DONE wins: termination is decided in the final round. Use `--selection hierarchical` for the earlier beam search or `--selection shortlist` for the lowest-call mode.

## Bounded live check

Earlier benchmark with four concurrent requests: `Say hello in one short sentence.` Both modes were capped at two output tokens with the same model and key.

| Mode | Text | Stop | HTTP calls | Reported input tokens | Time |
| --- | --- | --- | ---: | ---: | ---: |
| Hierarchical | `Hello` | DONE | 8 | 41,892 | 2.22 s |
| Tournament | `Hello.` | Two-token cap | 88 | 3,523,613 | 10.24 s |

This verifies live full-vocabulary selection, not answer quality or a speed advantage. The tournament did not reach a DONE decision before the test cap. Its longer response also contains more emitted tokens; the timings are single observations, not a controlled quality benchmark.

With the all-batches-concurrent default, a separate one-token run of the same greeting produced `Hello` in 2.2 seconds, using 44 HTTP calls and 1,761,787 reported input tokens. It hit the one-token cap. This is a single timing sample, not a guarantee; parallelism reduces elapsed waiting, not classification work or input usage.
