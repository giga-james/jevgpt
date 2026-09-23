"""Four stratified token samples, four parallel winners, one final decision."""
from .client import DONE


class ParallelSelection:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def samples(self, prompt, answer):
        vocab = self.vocabulary
        buckets = [[] for _ in range(4)]
        seen = set()

        def distribute(source, quota):
            added = [0] * 4
            cursor = 0
            for token in source:
                if token not in vocab.text or token in seen:
                    continue
                available = [i for i in range(4) if added[i] < quota and len(buckets[i]) < 254]
                if not available:
                    break
                index = next((i for i in range(cursor, 4) if i in available), available[0])
                buckets[index].append(token)
                seen.add(token)
                added[index] += 1
                cursor = (index + 1) % 4

        # Round-robin each stratum across the samples; never duplicate tokens.
        encoded = vocab.encode(answer)
        context = list(dict.fromkeys(list(reversed(encoded[-128:])) + vocab.encode(prompt)))
        distribute(context, 32)
        distribute(list(dict.fromkeys(vocab.common + vocab.fallback)), 64)
        suffix = encoded or vocab.encode("Assistant:")
        continuations = []
        for width in (3, 2, 1):
            continuations.extend(t for t, _ in vocab.followers[tuple(suffix[-width:])].most_common())
        distribute(continuations, 32)
        # This combines lexical matches and exploratory samples from the full vocab.
        distribute(vocab.expansion(prompt, answer, None, seen, limit=248), 62)
        remaining = [t for t in vocab.text if t not in seen]
        vocab.rng.shuffle(remaining)
        distribute(remaining, 64)
        distribute((t for t in remaining if t not in seen), 254)
        for bucket in buckets:
            vocab.rng.shuffle(bucket)
        if any(len(bucket) != 254 for bucket in buckets):
            raise ValueError("Four samples require at least 1,016 eligible tokens")
        return buckets

    def choose(self, client, state, on_decision=None):
        buckets = self.samples(state["user_message"], state["assistant_reply_so_far"])
        questions = [{f"t{t}": {"append_exact_text": self.vocabulary.text[t]} for t in bucket}
                     for bucket in buckets]
        rankings = client.rank_many(state, questions, (
            "Choose the best next literal text fragment to append to assistant_reply_so_far "
            "to answer user_message helpfully and grammatically. Preserve spaces and punctuation. "
            "Do not repeat text or add role labels. These are alternative candidates for ONE "
            "next-token position, not consecutive output tokens."
        ), separate_requests=True)
        if len(rankings) != 4:
            raise ValueError("Missing parallel draft results")
        finalists = {}
        for bucket, probabilities in zip(questions, rankings):
            if set(probabilities) != set(bucket):
                raise ValueError("Jev returned options outside the draft sample")
            winner = max(probabilities, key=probabilities.get)
            finalists[winner] = self.vocabulary.text[int(winner[1:])]
            if on_decision:
                on_decision(0, winner, probabilities[winner], len(bucket))
        # Only this comparison decides the emitted token. Do not compare local scores.
        choice, probability = client.choose(state, finalists)
        if choice != DONE and choice not in finalists:
            raise ValueError("Jev chose an option outside the finalists")
        if on_decision:
            on_decision(1, choice, probability, len(finalists) + 1)
        return choice, probability
