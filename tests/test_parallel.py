import json
from threading import Barrier, Lock

import httpx
import pytest

from jevgpt.chat import generate
from jevgpt.client import JevClient, DONE
from jevgpt.parallel import ParallelSelection
from jevgpt.vocabulary import Vocabulary


@pytest.fixture(scope="module")
def vocab():
    return Vocabulary(seed=42)


def test_samples_are_disjoint_stratified_and_cover_1016_tokens(vocab):
    buckets = ParallelSelection(vocab).samples("Tell me about astronomy", "The sky")
    assert [len(b) for b in buckets] == [254] * 4
    assert len({t for bucket in buckets for t in bucket}) == 1016
    assert set(vocab.encode("Tell me about astronomy")) <= {t for b in buckets for t in b}
    assert all(any(t in vocab.common for t in b) for b in buckets)
    assert all(t in vocab.text for b in buckets for t in b)


def test_four_requests_are_parallel_same_position_then_one_verifier(vocab):
    barrier = Barrier(4)
    lock = Lock()
    drafts_finished = 0
    finalist_keys = []
    draft_states = []
    final_states = []

    def respond(request):
        nonlocal drafts_finished
        data = json.loads(request.content)
        assert len(data["questions"]) == 1
        key, question = next(iter(data["questions"].items()))
        criteria = question["criteria"]
        if DONE not in criteria:
            assert len(criteria) == 254
            winner = next(iter(criteria))
            with lock:
                draft_states.append(data["state"])
                finalist_keys.append(winner)
            barrier.wait(timeout=5)  # Fails if drafts are accidentally serialized.
            with lock:
                drafts_finished += 1
        else:
            assert drafts_finished % 4 == 0
            assert len(criteria) == 5
            assert set(criteria) - {DONE} == set(finalist_keys[-4:])
            final_states.append(data["state"])
            winner = DONE if len(final_states) == 2 else list(criteria)[-2]
        return httpx.Response(200, json={"answers": {key: {
            "probabilities": {k: float(k == winner) for k in criteria}
        }}})

    client = JevClient("fake", transport=httpx.MockTransport(respond))
    streamed = []
    try:
        result = generate(client, vocab, "Hello", hierarchy=ParallelSelection(vocab),
                          on_token=lambda text, *_: streamed.append(text))
        assert client.calls == 10
        assert result.stop_reason == "done"
        assert len(result.tokens) == len(streamed) == 1
        assert all(s == final_states[0] for s in draft_states[:4])
        assert all(s == final_states[1] for s in draft_states[4:])
        assert final_states[0]["assistant_reply_so_far"] == ""
        assert final_states[1]["assistant_reply_so_far"] == result.text
    finally:
        client.close()


def test_missing_draft_fails_before_verification(vocab):
    class Missing:
        def rank_many(self, *args, **kwargs):
            return []

    with pytest.raises(ValueError, match="Missing"):
        ParallelSelection(vocab).choose(Missing(), {"user_message": "Hi", "assistant_reply_so_far": ""})
