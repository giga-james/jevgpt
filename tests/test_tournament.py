from collections import Counter
from types import SimpleNamespace

import pytest

from jevgpt.client import DONE
from jevgpt.tournament import Tournament
from jevgpt.vocabulary import Vocabulary
from jevgpt.chat import generate


def test_full_vocabulary_is_partitioned_once():
    vocab = Vocabulary()
    selector = Tournament(vocab)
    tokens = [int(k[1:]) for bucket in selector.buckets for k in bucket]
    assert Counter(tokens) == Counter({t: 1 for t in vocab.text})
    assert all(1 <= len(bucket) <= 255 for bucket in selector.buckets)
    assert len(selector.buckets) == 390
    assert selector.buckets == Tournament(vocab).buckets
    assert selector.buckets != Tournament(vocab, seed=1).buckets


def test_winners_are_recompared_instead_of_comparing_local_probabilities():
    vocab = SimpleNamespace(text={i: f" word{i}" for i in range(800)})
    selector = Tournament(vocab, bucket_size=2)
    visited = []
    rounds = []

    class Oracle:
        def rank_many(self, state, buckets, instructions):
            rounds.append(len(buckets))
            result = []
            for bucket in buckets:
                assert DONE not in bucket
                keys = list(bucket)
                visited.extend(keys)
                # Buckets have different strengths; every bucket must advance a winner.
                p = 0.99 if len(result) % 2 == 0 else 0.51
                result.append({keys[0]: p, keys[1]: 1-p})
            return result

        def choose(self, state, candidates):
            assert len(candidates) == 200
            return list(candidates)[-1], 0.8

    choice, _ = selector.choose(Oracle(), {})
    assert rounds == [400, 200]
    assert choice in visited
    assert len(set(visited[:800])) == 800


def test_feedback_and_done_only_append_final_token():
    vocab = SimpleNamespace(text={1: " hello", 2: " world", 3: "!"})
    tree = Tournament(vocab, bucket_size=2)
    states = []

    class Oracle:
        def rank_many(self, state, buckets, instructions):
            states.append(state.copy())
            return [{k: float(i == 0) for i, k in enumerate(b)} for b in buckets]

        def choose(self, state, candidates):
            return (DONE, 1.0) if state["assistant_reply_so_far"] else (next(iter(candidates)), 0.8)

    streamed = []
    result = generate(Oracle(), vocab, "Hi", hierarchy=tree,
                      on_token=lambda text, *_: streamed.append(text))
    assert result.stop_reason == "done"
    assert len(result.tokens) == len(streamed) == 1
    assert states[0]["assistant_reply_so_far"] == ""
    assert states[1]["assistant_reply_so_far"] == result.text


def test_invalid_and_missing_bucket_results_fail():
    tree = Tournament(SimpleNamespace(text={1: "a", 2: "b"}))
    for rankings in ([], [{"unknown": 1.0}]):
        client = SimpleNamespace(rank_many=lambda *args: rankings)
        with pytest.raises(ValueError):
            tree.choose(client, {})
