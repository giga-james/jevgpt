from types import SimpleNamespace

from jevgpt.chat import generate
from jevgpt.client import DONE
from jevgpt.speculative import Speculative
from jevgpt.vocabulary import Vocabulary


def fixture(drafts, verified):
    vocab = SimpleNamespace(
        text={1: "A", 2: "B", 3: "C", 4: "D", 5: " new", 6: " other"},
        shortlist=lambda *_: [1, 2, 3, 4],
        expansion=lambda *_: [5, 6],
    )

    class Client:
        def __init__(self):
            self.drafts = iter(drafts)
            self.questions = []
            self.states = []
            self.calls = 0

        def rank_criteria(self, state, criteria, instructions):
            self.calls += 1
            choice = next(self.drafts)
            return {key: float(key == choice) for key in criteria}

        def rank_questions(self, state, questions):
            self.calls += 1
            self.states.append(dict(state))
            self.questions.append(questions)
            choices = ["t5"] * len(questions) if len(self.questions) == 1 else verified
            return [{key: float(key == choice) for key in question["criteria"]}
                    for question, choice in zip(questions, choices)]

    return vocab, Client()


def test_four_drafts_acceptance_streams_only_verified_tokens():
    vocab, client = fixture(["t1", "t2", "t3", "t4"], ["t1", "t2", "t3", "t4"])
    selector = Speculative(vocab)
    streamed = []
    result = generate(client, vocab, "Hi", max_tokens=4, speculative=selector,
                      on_token=lambda text, *_: streamed.append(text))
    assert result.text == "ABCD"
    assert streamed == list("ABCD")
    assert result.stop_reason == "max_tokens"
    assert client.calls == 6
    assert selector.drafted == selector.accepted == 4
    assert all("assistant_reply_so_far" not in state for state in client.states)
    for questions in client.questions:
        assert [q["instructions"]["assistant_reply_so_far"] for q in questions] == ["", "A", "AB", "ABC"]
        assert all(len(q["criteria"]) <= 255 for q in questions)


def test_first_disagreement_discards_remaining_suffix_and_expands_vocabulary():
    vocab, client = fixture(["t1", "t2", "t3", "t4"], ["t1", "t5", "t3", "t4"])
    selector = Speculative(vocab)
    decisions = selector.block(client, {"user_message": "Hi", "assistant_reply_so_far": ""}, [], 4)
    assert [choice for choice, _ in decisions] == ["t1", "t5"]
    assert selector.accepted == 1
    assert "t5" not in {f"t{t}" for t in vocab.shortlist()}


def test_done_is_verified_and_can_be_overridden():
    vocab, client = fixture([DONE], ["t5"])
    selector = Speculative(vocab)
    assert selector.block(client, {"user_message": "Hi", "assistant_reply_so_far": ""}, [], 4) == [("t5", 1.0)]
    vocab, client = fixture(["t1", "t2", "t3", "t4"], [DONE, "t2", "t3", "t4"])
    result = generate(client, vocab, "Hi", max_tokens=4, speculative=Speculative(vocab))
    assert result.text == "" and result.stop_reason == "done"


def test_expansion_is_novel_and_preserves_exact_token_bytes():
    vocab = Vocabulary(seed=0)
    baseline = vocab.shortlist("Tell me about astronomy", [])
    extra = vocab.expansion("Tell me about astronomy", "", baseline[0], baseline)
    assert len(extra) == len(set(extra)) == 128
    assert not set(extra).intersection(baseline)
    assert all(t in vocab.text for t in extra)
    assert any(vocab.text[t].strip().casefold().startswith("ast") for t in extra)


def test_drafting_respects_remaining_output_budget():
    vocab, client = fixture(["t1", "t2"], ["t1", "t2"])
    result = generate(client, vocab, "Hi", max_tokens=2, speculative=Speculative(vocab))
    assert result.text == "AB"
    assert client.calls == 4
