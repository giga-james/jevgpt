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
    result = generate(Oracle(), vocab, "Hello", hierarchy=hierarchy,
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

    assert hierarchy.choose(Stop(), {}) == (DONE, 1.0)

    class Invalid:
        def choose_criteria(self, *args):
            return "g999", 1.0

    with pytest.raises(ValueError, match="outside"):
        hierarchy.choose(Invalid(), {})


def test_single_leaf_and_whitespace_are_exact():
    vocab = SimpleNamespace(text={1: " a", 2: "a", 3: "\n"}, frequency=Counter())
    tree = Hierarchy(vocab)
    assert tree.options(tree.root, True)["t1"] == {"append_exact_text": " a"}
    assert tree.options(tree.root, True)["t3"] == {"append_exact_text": "\n"}
    with pytest.raises(ValueError):
        Hierarchy(vocab, fanout=1)
