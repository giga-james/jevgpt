# JevGPT experiment log

The question behind JevGPT: can a classifier become a chatbot if we repeatedly ask it to choose the next text fragment?

This log records the path from a small heuristic shortlist to the current **Speculative** mode: four parallel candidate drafts for the same position, followed by one verifier. It includes unsuccessful approaches and implementation corrections, not just the current design.

These were bounded exploratory runs with Jev 1.13.0, a tiny bundled dialogue corpus, and `cl100k_base`. Runs used different output caps and revisions, and model responses varied. Timings are individual observations, not a controlled benchmark or latency guarantee. Input counts are API-reported usage. A more grammatical answer is not necessarily a correct answer.

## 1. Establish the heuristic baseline

[Implementation: c45d58c](https://github.com/giga-james/jevgpt/commit/c45d58c)

**Idea:** present 254 candidate tokens plus DONE, choose one, append it, and repeat. Construct candidates from basic characters, corpus frequency, prompt/recent-output tokens, n-gram continuations, and exploration. One API call selects each token or DONE.

**Observed:** a greeting produced ` Hello`, then DONE. The sky-blue prompt produced ` It is is blue.` in another run. The loop worked, but the candidate shortlist could exclude useful continuations. Live responses also exposed rounded probabilities summing to 0.99; validation was adjusted to allow rounding.

**Next question:** would wider vocabulary access help more than a better shortlist?

## 2. Route through the vocabulary hierarchy

[Implementation: 6b4b796](https://github.com/giga-james/jevgpt/commit/6b4b796)

**Idea:** make all 99,412 eligible text tokens reachable. Sort fragments lexicographically, partition into up to 32 groups per level, and ask Jev to choose a group, subgroup, then exact token. Group descriptions contained boundaries and six examples. This normally required three calls per emitted token.

**Observed:** one sky-blue run produced ` It Its Is It It It blue`, then DONE, using 22 calls. Broader reach did not translate into coherence.

**Lesson:** choosing an alphabetical range is an awkward task. Once the router picks the wrong branch, the desired token is unavailable. Group examples can also bias routing. We had improved reachability, not demonstrated better generation.

## 3. Keep multiple branches and compare real tokens

[Implementation: 5cce4f5](https://github.com/giga-james/jevgpt/commit/5cce4f5)

**Idea:** retain up to three branches, select up to 16 candidates per surviving leaf, and make a fresh final comparison among up to 48 fragments plus DONE. Routing-score products were used only as search heuristics.

**Observed:** a bounded comparison yielded:

| Prompt | Greedy hierarchy | Multi-branch hierarchy | Calls, greedy / multi-branch |
| --- | --- | --- | ---: |
| Why is the sky blue? Answer in one short sentence. | ` It Its Is It It It It blue` | ` Because the Air air air blocks blue` | 25 / 64 |
| Say hello in one short sentence. | `Hello` | `Hello` | 4 / 16 |

All four runs selected DONE. The second sky answer was more relevant but still false and repetitive. The search now needed eight sequential calls per generation step before batching.

**Lesson:** keeping alternatives reduces early lock-in, but it is expensive and does not repair the classifier's language-generation weaknesses.

## 4. Batch independent hierarchy decisions

[Implementation: 0065a6d](https://github.com/giga-james/jevgpt/commit/0065a6d)

**Idea:** send sibling questions together. Keep the same search but reduce eight round trips to four stages: root, subgroups, leaves, final comparison.

**Observed:** on an eight-output-token sky-prompt test, serial ranking used 64 requests and 11.86 seconds; batching used 32 requests and 6.55 seconds. Both hit the cap. That is about 45% less elapsed time in this sample, without shrinking the beam.

**Lesson:** eliminate unnecessary serial waiting before reducing coverage. This improved execution, not answer quality.

## 5. Evaluate every vocabulary bucket

[Tournament: bec4d28](https://github.com/giga-james/jevgpt/commit/bec4d28) · [Full batch concurrency: 12a8a64](https://github.com/giga-james/jevgpt/commit/12a8a64)

**Idea:** stop asking Jev to guess a range. Evaluate every token in 390 disjoint buckets, promote one winner per bucket, then run two semifinals and a final with DONE. Every bucket addresses the same next-token position. Compare winners directly; probabilities from different buckets are not comparable.

**Observed:** with four concurrent requests, a two-token greeting test produced `Hello.` in 10.24 seconds, using 88 calls and 3,523,613 reported input tokens. It hit the cap rather than selecting DONE. In the same comparison, hierarchy produced `Hello` and DONE in 2.22 seconds with eight calls and 41,892 input tokens.

Launching all first-round batches concurrently later produced `Hello` in a separate one-token test: 2.2 seconds, 44 calls, and 1,761,787 input tokens. That run also stopped at its cap.

**Lesson:** exhaustive evaluation is costly even when parallelized. Concurrency reduces waiting; it does not remove classification work or input usage. See the [tournament details](tournament.md).

## 6. Try four consecutive drafts—then correct the interpretation

[Superseded implementation: 86e5d6e](https://github.com/giga-james/jevgpt/commit/86e5d6e)

**Idea as implemented:** draft four consecutive tokens using the heuristic shortlist, batch-rank additional vocabulary candidates at each position, then verify. Accept the matching prefix and discard the suffix after a disagreement.

This was a misunderstanding of the intended four drafts: they were supposed to be competing candidates for one position, not consecutive output tokens.

**Observed:** with a 12-token cap on the sky question, original shortlist returned ` It is is appears appears is the` in 1.69 seconds and eight calls. The sequential draft experiment returned ` The sky is blue.` in 4.48 seconds and 23 calls. Both selected DONE. Only two of 15 drafted decisions were accepted; counts include DONE and discarded suffix work.

**Lesson:** the sentence was more grammatical but did not explain why. Drafting future tokens sequentially and throwing most away added latency. This implementation was removed, rather than retained under the same name.

## 7. Current: four parallel stratified samples

[Implementation: e9a2b51](https://github.com/giga-james/jevgpt/commit/e9a2b51) · [README naming: f3922da](https://github.com/giga-james/jevgpt/commit/f3922da)

**Intended approach:** sample four disjoint sets of 254 tokens from heuristic strata, giving 1,016 distinct candidates. Run four Jev calls concurrently against identical context. Each supplies one candidate for the same next-token position. A fifth call selects among the four winners and DONE.

**Observed:** the sky prompt produced `The sky is blue ForDue to the the the light of` before hitting the 12-token cap: 60 calls, 323,909 reported input tokens, and 8.0 seconds, approximately 0.67 seconds per emitted token. The output remains incomplete and repetitive. This is not evidence of a reliable quality gain.

**What changed:** five total calls now have only two sequential stages. There are no future-token drafts, acceptance checks, or discarded suffixes. A concurrency test checks that all four draft calls overlap; another check ensures only the verifier's selected token enters the reply. The CLI stays a plain chatbot, with diagnostics opt-in.

We call this **Speculative** mode, inspired by draft-and-verify approaches such as [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192). It is a sampled candidate tournament, not an implementation of that paper's distribution-preserving decoding algorithm. See [current algorithm and sampling quotas](parallel.md).

## What we have learned—and what remains open

- More vocabulary coverage has not reliably improved answers. Candidate quality and Jev's ability to judge continuations both matter.
- Fewer serial request stages help latency. More concurrency alone does not reduce usage.
- Recompare finalists directly instead of treating bucket-local probabilities as global token scores.
- Keep the one-call shortlist as the baseline. Extra machinery must justify its cost against that baseline.

Useful next experiments would use a fixed prompt set, repeated runs, consistent output caps, time to first emitted token, total latency, input usage, and a separate assessment of correctness and repetition. Then vary one factor at a time: sampling quotas, corpus quality, number of parallel samples, or batched versus separate draft requests. The current observations do not establish an optimal configuration.

[Back to README](../README.md)
