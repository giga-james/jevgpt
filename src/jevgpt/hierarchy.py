"""Bounded, exhaustive lexicographic routing over eligible text tokens."""
from dataclasses import dataclass
from os.path import commonprefix

from .client import DONE


@dataclass
class Node:
    tokens: tuple[int, ...]
    children: tuple = ()


class Hierarchy:
    def __init__(self, vocabulary, fanout=32, leaf_size=128):
        if not 2 <= fanout <= 254 or not 1 <= leaf_size <= 254:
            raise ValueError("Invalid hierarchy fanout or leaf size")
        self.vocabulary = vocabulary
        ordered = tuple(sorted(vocabulary.text, key=lambda t: vocabulary.text[t]))
        if not ordered:
            raise ValueError("Vocabulary is empty")

        def build(tokens):
            if len(tokens) <= leaf_size:
                return Node(tokens)
            width = (len(tokens) + fanout - 1) // fanout
            return Node(tokens, tuple(build(tokens[i:i + width]) for i in range(0, len(tokens), width)))

        self.root = build(ordered)

    def options(self, node, root=False):
        text = self.vocabulary.text
        if node.children:
            criteria = {}
            for i, child in enumerate(node.children):
                first, last = text[child.tokens[0]], text[child.tokens[-1]]
                examples = sorted(child.tokens, key=lambda t: (-self.vocabulary.frequency[t], t))[:6]
                criteria[f"g{i}"] = {
                    "shared_prefix": commonprefix([first, last]),
                    "first_fragment": first,
                    "last_fragment": last,
                    "examples_not_exhaustive": [text[t] for t in examples],
                    "token_count": len(child.tokens),
                }
        else:
            criteria = {f"t{t}": {"append_exact_text": text[t]} for t in node.tokens}
        if root:
            criteria[DONE] = "The assistant reply is complete. End without adding text."
        return criteria

    def choose(self, client, state, on_decision=None):
        node = self.root
        depth = 0
        while True:
            root = node is self.root
            criteria = self.options(node, root)
            instructions = (
                "Continue a helpful, concise assistant reply to user_message, appending after "
                "assistant_reply_so_far. Preserve literal spaces and punctuation. Do not repeat "
                "the user or add role labels. "
            )
            if node.children:
                instructions += (
                    "Choose the GROUP containing the best next text fragment. Groups are sorted "
                    "lexicographically by exact text (including leading whitespace), with inclusive "
                    "first/last boundaries. Examples are illustrative, not the entire group. "
                    "Choose based on the continuation you want, not on the number of tokens. "
                    "This is a routing decision; do not append the group label or any example."
                )
            else:
                instructions += "Choose the exact next text fragment to append."
            if root:
                instructions += " Choose DONE only if the reply is already complete."
            choice, probability = client.choose_criteria(state, criteria, instructions)
            if choice not in criteria:
                raise ValueError("Jev selected an option outside the current hierarchy node")
            if on_decision:
                on_decision(depth, choice, probability, len(criteria))
            if choice == DONE or not node.children:
                # Return the leaf's local probability, not a global token probability.
                return choice, probability
            node = node.children[int(choice[1:])]
            depth += 1
