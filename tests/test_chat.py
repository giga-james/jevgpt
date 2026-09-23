import json

import httpx
import pytest

from jevgpt.chat import generate
from jevgpt.client import DONE, JevClient
from jevgpt.vocabulary import Vocabulary


@pytest.fixture(scope="module")
def vocab():
    return Vocabulary(seed=42)


def test_vocabulary_roundtrip_and_shortlist(vocab):
    choices = vocab.shortlist("Hello, zebras!", [])
    assert len(choices) == len(set(choices)) == 254
    assert set(vocab.fallback) <= set(choices)
    assert set(vocab.encode("Hello, zebras!")) <= set(choices)
    for token in choices:
        assert vocab.text[token].encode() == vocab.encoding.decode_single_token_bytes(token)
    for special in vocab.encoding.special_tokens_set:
        assert vocab.encoding.encode_single_token(special) not in vocab.text


def test_corpus_continuation_is_available():
    vocab = Vocabulary("alpha zebra\n" * 4)
    prefix = vocab.encode("alpha")
    expected = vocab.encode("alpha zebra")[1]
    assert expected in vocab.shortlist("Continue", prefix)


def test_generation_feeds_back_full_answer_and_stops(vocab):
    states = []
    pieces = iter(["H", "i", None])

    class Fake:
        def choose(self, state, candidates):
            states.append(state)
            fragment = next(pieces)
            return (DONE if fragment is None else f"t{vocab.encode(fragment)[0]}", 0.8)

    streamed = []
    result = generate(Fake(), vocab, "Hello", on_token=lambda text, *_: streamed.append(text))
    assert result.text == "Hi"
    assert result.stop_reason == "done"
    assert streamed == ["H", "i"]
    assert [s["assistant_reply_so_far"] for s in states] == ["", "H", "Hi"]


def test_bounds_and_repetition(vocab):
    class Fake:
        def choose(self, state, candidates):
            return f"t{vocab.encode('a')[0]}", 1

    assert generate(Fake(), vocab, "Hi", max_tokens=2).stop_reason == "max_tokens"
    assert generate(Fake(), vocab, "Hi", max_tokens=20).stop_reason == "repetition"
    with pytest.raises(ValueError):
        generate(Fake(), vocab, "x" * 8001)
    with pytest.raises(ValueError):
        generate(Fake(), vocab, " ")


def test_api_contract_and_argmax():
    def respond(request):
        assert request.url == "https://api.typesafe.ai/v1/systemone"
        payload = json.loads(request.content)
        question = payload["questions"]["next_token"]
        assert question["criteria"]["t1"] == {"append_exact_text": " hello"}
        assert DONE in question["criteria"]
        return httpx.Response(200, json={"answers": {"next_token": {
            "choice": "DONE", "probabilities": {"t1": 0.9, "DONE": 0.09},
        }}, "usage": {"input_tokens": 123}})

    client = JevClient("fake", transport=httpx.MockTransport(respond))
    try:
        assert client.choose({}, {"t1": " hello"}) == ("t1", 0.9)
        assert client.calls == 1
        assert client.input_tokens == 123
    finally:
        client.close()


def test_api_retries_and_rejects_invalid_choices(monkeypatch):
    monkeypatch.setattr("jevgpt.client.time.sleep", lambda _: None)
    responses = iter([
        httpx.Response(429),
        httpx.Response(200, json={"answers": {"next_token": {
            "probabilities": {"unknown": 1.0},
        }}}),
    ])
    client = JevClient("fake", transport=httpx.MockTransport(lambda _: next(responses)))
    try:
        with pytest.raises(ValueError, match="invalid probability"):
            client.choose({}, {"t1": " hello"})
        assert client.calls == 2
    finally:
        client.close()


def test_unlimited_generation_reaches_done_beyond_default_cap(vocab):
    sequence = iter(vocab.encode(' '.join(str(i) for i in range(150))) + [None])

    class Draft:
        def choose(self, client, state, on_decision):
            token = next(sequence)
            return (DONE if token is None else f't{token}', 1.0)

    result = generate(None, vocab, 'Count', max_tokens=0, hierarchy=Draft())
    assert len(result.tokens) > 128
    assert result.stop_reason == 'done'


def test_unlimited_generation_retains_repetition_guard(vocab):
    class Fake:
        def choose(self, state, candidates):
            return f"t{vocab.encode('a')[0]}", 1.0

    assert generate(Fake(), vocab, 'Hi', max_tokens=0).stop_reason == 'repetition'
