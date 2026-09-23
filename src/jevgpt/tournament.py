"""Evaluate every eligible token, then compare bucket winners directly."""
import json
import random

from .client import DONE


class Tournament:
    instructions = (
        "Choose the best next literal text fragment to append to assistant_reply_so_far "
        "in a helpful, grammatical answer to user_message. Each option's value is the exact "
        "fragment, including spaces and punctuation. Do not repeat text or add role labels. "
        "These options are competing continuations for ONE position, not a sequence."
    )

    def __init__(self, vocabulary, seed=0, bucket_size=255):
        if not 2 <= bucket_size <= 255:
            raise ValueError("Bucket size must be between 2 and 255")
        self.vocabulary = vocabulary
        self.bucket_size = bucket_size
        tokens = list(vocabulary.text)
        if not tokens:
            raise ValueError("Vocabulary is empty")
        random.Random(seed).shuffle(tokens)
        self.buckets = self._partition(tokens)

    def _partition(self, tokens):
        buckets = []
        bucket = {}
        size = 2
        for token in tokens:
            key, value = f"t{token}", self.vocabulary.text[token]
            entry_size = len(json.dumps({key: value}, ensure_ascii=False).encode())
            if entry_size > 12000:
                raise ValueError("Token text exceeds the tournament bucket budget")
            if bucket and (len(bucket) == self.bucket_size or size + entry_size > 12000):
                buckets.append(bucket)
                bucket, size = {}, 2
            bucket[key] = value
            size += entry_size
        if bucket:
            buckets.append(bucket)
        return buckets

    def choose(self, client, state, on_decision=None):
        buckets = self.buckets
        round_index = 0
        while True:
            rankings = client.rank_many(state, buckets, self.instructions)
            if len(rankings) != len(buckets):
                raise ValueError("Missing tournament bucket results")
            winners = []
            for bucket, probabilities in zip(buckets, rankings):
                if set(probabilities) != set(bucket):
                    raise ValueError("Jev returned options outside the tournament bucket")
                winner = max(probabilities, key=probabilities.get)
                winners.append(int(winner[1:]))
                if on_decision:
                    on_decision(round_index, winner, probabilities[winner], len(bucket))
            # Never compare probabilities across buckets: re-evaluate the winners.
            if len(winners) <= 254:
                candidates = {f"t{t}": self.vocabulary.text[t] for t in winners}
                choice, probability = client.choose(state, candidates)
                if choice != DONE and choice not in candidates:
                    raise ValueError("Jev selected an option outside the tournament finalists")
                if on_decision:
                    on_decision(round_index + 1, choice, probability, len(candidates) + 1)
                return choice, probability
            buckets = self._partition(winners)
            round_index += 1

    def summary(self):
        return {"eligible_tokens": len(self.vocabulary.text), "buckets": len(self.buckets),
                "largest_bucket": max(map(len, self.buckets)),
                "selection": "all buckets evaluated, then winners re-compared with DONE in final"}
