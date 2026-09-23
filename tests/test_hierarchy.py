import json
from collections import Counter
from types import SimpleNamespace

import pytest

from jevgpt.chat import generate
from jevgpt.client import DONE
from jevgpt.hierarchy import Hierarchy
from jevgpt.vocabulary import Vocabulary


@pytest.fixture(scope="module")
def hierarchy():
    return Hierarchy(Vocabulary())


def test_full_vocabulary_coverage_and_bounded_nodes(hierarchy):
    leaves = []
    depths = []

    def visit(node, depth=0):
        options = hierarchy.options(node, node is hierarchy.root)
        assert 1 <= len(options) <= 255
        assert (DONE in options) == (depth == 0)
        assert len(json.dumps(options, ensure_ascii=False).encode()) < 14000
        if node.children:
            assert tuple(t for child in node.children for t in child.tokens) == node.tokens
            for child in node.children:
                visit(child, depth + 1)
        else:
            leaves.extend(node.tokens)
            depths.append(depth)

    visit(hierarchy.root)
    assert Counter(leaves) == Counter({t: 1 for t in hierarchy.vocabulary.text})
    assert max(depths) <= 3


def test_route_to_token_outside_shortlist_and_feedback(hierarchy):
    vocab = hierarchy.vocabulary
    excluded = set(vocab.shortlist("Hello", []))
    target = next(t for t in vocab.text if t not in excluded and len(vocab.text[t]) > 4)
    node = hierarchy.root
    states = []

    class Oracle:
        def choose_criteria(self, state, criteria, instructions):
            nonlocal node
            states.append(dict(state))
            if state["assistant_reply_so_far"]:
                assert DONE in criteria
                return DONE, 0.9
            if node.children:
                index = next(i for i, child in enumerate(node.children) if target in child.tokens)
                node = node.children[index]
                return f"g{index}", 0.8
            return f"t{target}", 0.7

    events = []
    result = generate(Oracle(), vocab, "Hello", hierarchy=SimpleNamespace(choose=hierarchy.choose_greedy),
                      on_decision=lambda *event: events.append(event))
    assert result.tokens == [target]
    assert result.text == vocab.text[target]
    assert result.stop_reason == "done"
    assert all(s["assistant_reply_so_far"] == "" for s in states[:-1])
    assert states[-1]["assistant_reply_so_far"] == result.text
    assert events[-1][1] == DONE


def test_root_done_and_invalid_route(hierarchy):
    class Stop:
        def choose_criteria(self, *args):
            return DONE, 1.0

    assert hierarchy.choose_greedy(Stop(), {}) == (DONE, 1.0)

    class Invalid:
        def choose_criteria(self, *args):
            return "g999", 1.0

    with pytest.raises(ValueError, match="outside"):
        hierarchy.choose_greedy(Invalid(), {})


def test_single_leaf_and_whitespace_are_exact():
    vocab = SimpleNamespace(text={1: " a", 2: "a", 3: "\n"}, frequency=Counter())
    tree = Hierarchy(vocab)
    assert tree.options(tree.root, True)["t1"] == {"append_exact_text": " a"}
    assert tree.options(tree.root, True)["t3"] == {"append_exact_text": "\n"}
    with pytest.raises(ValueError):
        Hierarchy(vocab, fanout=1)


def test_beam_recovers_second_branch_and_final_choice_controls_output():
    vocab = SimpleNamespace(text={i: chr(97 + i) for i in range(8)}, frequency=Counter())
    tree = Hierarchy(vocab, fanout=2, leaf_size=2, beam_width=2, per_leaf=1)
    states = []

    class Ranked:
        def rank_criteria(self, state, criteria, instructions):
            states.append(state.copy())
            assert DONE not in criteria
            assert len(criteria) <= 255
            keys = list(criteria)
            # First branch 0.6; second 0.4. Concentrated child scores preserve both parents.
            values = [0.6, 0.4] if len(states) == 1 else [0.9, 0.1]
            return dict(zip(keys, values))

        def choose(self, state, candidates):
            states.append(state.copy())
            assert set(candidates) == {"t0", "t4"}
            return "t4", 0.9

    state = {"user_message": "Hi", "assistant_reply_so_far": ""}
    assert tree.choose(Ranked(), state) == ("t4", 0.9)
    assert len(states) == 6  # root + two branches + two leaves + final
    assert all(s == state for s in states)


def test_beam_done_only_in_final_comparison_and_candidate_cap():
    vocab = SimpleNamespace(text={i: str(i) for i in range(300)}, frequency=Counter())
    tree = Hierarchy(vocab)

    class Ranked:
        def rank_criteria(self, state, criteria, instructions):
            assert DONE not in criteria
            return {key: 1 / len(criteria) for key in criteria}

        def choose(self, state, candidates):
            assert 1 <= len(candidates) <= 48
            return DONE, 1

    assert tree.choose(Ranked(), {}) == (DONE, 1)
    with pytest.raises(ValueError):
        Hierarchy(vocab, beam_width=8, per_leaf=32)
