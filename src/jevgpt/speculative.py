"""Heuristic Jev drafts with batched vocabulary expansion and verification.

This is greedy application-level speculation, not distribution-preserving sampling.
"""
from .client import DONE


INSTRUCTIONS = (
    "Choose the best next literal fragment to append to assistant_reply_so_far, "
    "answering user_message helpfully and grammatically. Preserve spaces and punctuation. "
    "Do not repeat text or add role labels. Choose DONE only when the reply is complete."
)


class Speculative:
    def __init__(self, vocabulary, draft_length=4):
        if not 1 <= draft_length <= 6:
            raise ValueError("Draft length must be between 1 and 6")
        self.vocabulary = vocabulary
        self.draft_length = draft_length
        self.drafted = 0
        self.accepted = 0
        self.blocks = 0

    def _question(self, prefix, criteria):
        return {"type": "choice", "instructions": {
            "question": INSTRUCTIONS + " Use the assistant_reply_so_far in THIS question as the exact prefix.",
            "assistant_reply_so_far": prefix,
        }, "criteria": criteria}

    def block(self, client, state, tokens, remaining, on_decision=None):
        vocab = self.vocabulary
        prefix = state["assistant_reply_so_far"]
        draft_tokens = list(tokens)
        drafts = []
        for _ in range(min(self.draft_length, remaining)):
            candidates = vocab.shortlist(state["user_message"], draft_tokens)
            criteria = {f"t{t}": {"append_exact_text": vocab.text[t]} for t in candidates}
            criteria[DONE] = "The reply is complete; append nothing."
            probabilities = client.rank_criteria({**state, "assistant_reply_so_far": prefix},
                                                 criteria, INSTRUCTIONS)
            choice = max(probabilities, key=probabilities.get)
            if choice not in criteria:
                raise ValueError("Draft chose an unknown token")
            # Preserve several baseline alternatives for the final joint comparison.
            top = sorted(probabilities, key=probabilities.get, reverse=True)[:8]
            drafts.append((prefix, candidates, choice, {k: criteria[k] for k in top}))
            self.drafted += 1
            if choice == DONE:
                break
            token = int(choice[1:])
            draft_tokens.append(token)
            prefix += vocab.text[token]

        # Common state deliberately excludes the draft suffix. Each question only sees
        # its own preceding prefix, never its proposed token or any later draft tokens.
        common = {k: v for k, v in state.items() if k != "assistant_reply_so_far"}
        expansion_questions = []
        for prefix, candidates, choice, _ in drafts:
            extra = vocab.expansion(state["user_message"], prefix,
                                    None if choice == DONE else int(choice[1:]), candidates)
            expansion_questions.append(self._question(prefix, {
                f"t{t}": {"append_exact_text": vocab.text[t]} for t in extra
            }))
        expanded = client.rank_questions(common, expansion_questions)
        if len(expanded) != len(drafts):
            raise ValueError("Missing expansion results")
        verification_questions = []
        for (prefix, _, _, baseline), question, probabilities in zip(drafts, expansion_questions, expanded):
            finalists = dict(baseline)
            for key in sorted(probabilities, key=probabilities.get, reverse=True)[:8]:
                finalists[key] = question["criteria"][key]
            finalists[DONE] = "The reply is complete; append nothing."
            verification_questions.append(self._question(prefix, finalists))
        verified = client.rank_questions(common, verification_questions)
        if len(verified) != len(drafts):
            raise ValueError("Missing verification results")
        self.blocks += 1
        result = []
        for index, ((_, _, draft, _), question, probabilities) in enumerate(zip(drafts, verification_questions, verified)):
            choice = max(probabilities, key=probabilities.get)
            if choice not in question["criteria"]:
                raise ValueError("Verifier chose an unknown token")
            result.append((choice, probabilities[choice]))
            if on_decision:
                on_decision(index, choice, probabilities[choice], len(question["criteria"]))
            if choice == draft:
                self.accepted += 1
            # Never accept a later token whose conditioning prefix has just changed.
            if choice == DONE or choice != draft:
                break
        return result
