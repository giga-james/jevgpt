from collections import Counter, defaultdict
from importlib.resources import files
import random
import re

import tiktoken


class Vocabulary:
    def __init__(self, corpus: str | None = None, seed: int = 0):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.rng = random.Random(seed)
        special = {self.encoding.encode_single_token(s) for s in self.encoding.special_tokens_set}
        self.text = {}
        for token in range(self.encoding.max_token_value + 1):
            if token in special:
                continue
            try:
                value = self.encoding.decode_single_token_bytes(token).decode("utf-8")
            except (KeyError, UnicodeDecodeError):
                continue
            if value and all(c.isprintable() or c in "\n\r\t" for c in value):
                self.text[token] = value
        if corpus is None:
            corpus = files("jevgpt").joinpath("corpus.txt").read_text()
        tokens = self.encode(corpus)
        self.frequency = Counter(t for t in tokens if t in self.text)
        self.common = [t for t, _ in self.frequency.most_common()]
        self.followers = defaultdict(Counter)
        for i, token in enumerate(tokens):
            if token in self.text:
                for width in range(1, min(3, i) + 1):
                    self.followers[tuple(tokens[i - width:i])][token] += 1
        # Keep basic letters and punctuation reachable even with a tiny corpus.
        # Encode characters separately so BPE cannot merge adjacent characters.
        self.fallback = list(dict.fromkeys(t for c in "".join(chr(i) for i in range(32, 127)) + "\n\t" for t in self.encode(c)))

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text, disallowed_special=())

    def expansion(self, prompt, answer, draft, excluded, limit=128):
        """Retrieve novel lexical alternatives plus exploration, without API calls."""
        if not hasattr(self, "prefix_index"):
            self.prefix_index = defaultdict(list)
            for token, text in self.text.items():
                word = text.strip().casefold()
                if len(word) >= 3 and word.isalpha():
                    self.prefix_index[word[:3]].append(token)
        words = re.findall(r"[^\W\d_]+", prompt + " " + answer[-256:], re.UNICODE)
        if draft is not None:
            words.insert(0, self.text[draft].strip())
        pool = set()
        exact = {word.casefold() for word in words}
        for word in words:
            pool.update(self.prefix_index.get(word.casefold()[:3], ()))
        ranked = sorted(pool - set(excluded), key=lambda t: (
            self.text[t].strip().casefold() not in exact, -self.frequency[t], t))
        selected = ranked[:limit * 3 // 4]
        seen = set(excluded) | set(selected)
        remaining = [t for t in self.text if t not in seen]
        selected.extend(self.rng.sample(remaining, min(limit - len(selected), len(remaining))))
        return selected

    def shortlist(self, prompt: str, answer: list[int], limit: int = 254) -> list[int]:
        if not 1 <= limit <= 254:
            raise ValueError("Candidate limit must be between 1 and 254")
        selected = []
        seen = set()

        def add(items, quota):
            added = 0
            for token in items:
                if len(selected) >= limit or added >= quota:
                    break
                if token in self.text and token not in seen:
                    selected.append(token)
                    seen.add(token)
                    added += 1

        add(self.fallback, 97)
        add(self.common, 31)
        # Recent answer first, then the current user prompt; do not fill from old chat.
        context = list(reversed(answer[-128:])) + self.encode(prompt)
        add(context, 64)
        suffix = answer if answer else self.encode("Assistant:")
        continuations = []
        for width in (3, 2, 1):
            continuations.extend(t for t, _ in self.followers[tuple(suffix[-width:])].most_common())
        add(continuations, 46)
        # Weighted sampling without replacement from the corpus vocabulary.
        pool = [t for t in self.common if t not in seen]
        for _ in range(min(16, len(pool))):
            token = self.rng.choices(pool, weights=[self.frequency[t] for t in pool])[0]
            add([token], 1)
            pool.remove(token)
        add(self.common, limit)
        # Seeded uniform fallback makes tokens absent from the corpus reachable.
        pool = [t for t in self.text if t not in seen]
        add(self.rng.sample(pool, min(limit - len(selected), len(pool))), limit)
        self.rng.shuffle(selected)
        return selected
