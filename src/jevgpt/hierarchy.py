"""Bounded, exhaustive lexicographic routing over eligible text tokens."""
from dataclasses import dataclass
from os.path import commonprefix

from .client import DONE


@dataclass
class Node:
    tokens: tuple[int, ...]
    children: tuple = ()


class Hierarchy:
    def __init__(self, vocabulary, fanout=32, leaf_size=128, beam_width=3, per_leaf=16):
        if not 2 <= fanout <= 254 or not 1 <= leaf_size <= 254:
            raise ValueError("Invalid hierarchy fanout or leaf size")
        self.vocabulary = vocabulary
        if not 1 <= beam_width <= 8 or not 1 <= per_leaf <= 254 // beam_width:
            raise ValueError("Beam candidates must fit within 254 options")
        self.beam_width = beam_width
        self.per_leaf = per_leaf
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
        """Keep a bounded beam, then compare explicit fragments in a fresh Choice."""
        beam = [(self.root, 1.0)]
        depth = 0
        while any(node.children for node, _ in beam):
            expanded = []
            for node, score in beam:
                if not node.children:
                    expanded.append((node, score))
                    continue
                criteria = self.options(node)
                probabilities = client.rank_criteria(state, criteria, (
                    "Continue the assistant reply to user_message after assistant_reply_so_far. "
                    "Rank groups by which contains the best next literal text fragment. Groups "
                    "are sorted lexicographically, including spaces and capitalization, with "
                    "inclusive first/last boundaries. Examples are not exhaustive. Consider "
                    "the continuation needed, not group size. Do not output group labels."
                ))
                if set(probabilities) != set(criteria):
                    raise ValueError("Jev returned options outside the hierarchy node")
                total = sum(probabilities.values())
                for key in sorted(probabilities, key=probabilities.get, reverse=True)[:self.beam_width]:
                    expanded.append((node.children[int(key[1:])], score * probabilities[key] / total))
                    if on_decision:
                        on_decision(depth, key, probabilities[key], len(criteria))
            # Products are routing heuristics, not calibrated token probabilities.
            beam = sorted(expanded, key=lambda item: item[1], reverse=True)[:self.beam_width]
            depth += 1

        candidates = {}
        for node, _ in beam:
            criteria = self.options(node)
            probabilities = client.rank_criteria(state, criteria, (
                "Rank these exact literal fragments as the next continuation of "
                "assistant_reply_so_far answering user_message. Preserve spaces and punctuation. "
                "Prefer a helpful grammatical continuation without repetition or role labels."
            ))
            if set(probabilities) != set(criteria):
                raise ValueError("Jev returned options outside the hierarchy leaf")
            for key in sorted(probabilities, key=probabilities.get, reverse=True)[:self.per_leaf]:
                candidates[key] = self.vocabulary.text[int(key[1:])]
                if on_decision:
                    on_decision(depth, key, probabilities[key], len(criteria))
        # Compare candidates from different leaves together, without routing scores.
        choice, probability = client.choose(state, candidates)
        if choice != DONE and choice not in candidates:
            raise ValueError("Jev selected an option outside the final candidates")
        if on_decision:
            on_decision(depth + 1, choice, probability, len(candidates) + 1)
        return choice, probability

    def choose_greedy(self, client, state, on_decision=None):
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
