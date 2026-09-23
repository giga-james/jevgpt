import json
import math
import time

import httpx

DONE = "DONE"


class JevClient:
    def __init__(self, api_key: str, model: str = "jev-1.13.0", transport=None):
        self.model = model
        self.input_tokens = 0
        self.calls = 0
        self.http = httpx.Client(
            base_url="https://api.typesafe.ai", timeout=60,
            headers={"Authorization": f"Bearer {api_key}"}, transport=transport,
        )

    def close(self):
        self.http.close()

    def choose(self, state: dict, candidates: dict[str, str]) -> tuple[str, float]:
        criteria = {k: {"append_exact_text": v} for k, v in candidates.items()}
        criteria[DONE] = "The assistant reply is complete. End the reply without appending text."
        return self.choose_criteria(state, criteria, (
            "Continue the assistant reply to the current user message. Choose the single "
            "text fragment that should be appended next to assistant_reply_so_far. "
            "Fragments are literal text, including leading spaces and punctuation. "
            "Build a helpful, concise, grammatical reply. Do not repeat the user message "
            "or add role labels. Choose DONE only when the reply is complete."
        ))

    def choose_criteria(self, state: dict, criteria: dict, instructions: str):
        probabilities = self.rank_criteria(state, criteria, instructions)
        choice = max(probabilities, key=probabilities.get)
        return choice, probabilities[choice]

    def rank_criteria(self, state: dict, criteria: dict, instructions: str):
        return self._rank_questions(state, {
            "next_token": {"type": "choice", "instructions": instructions, "criteria": criteria}
        })["next_token"]

    def rank_many(self, state: dict, criteria_list: list[dict], instructions: str):
        """Evaluate independent questions in one round trip, with bounded payloads."""
        results = {}
        pending = {}
        for index, criteria in enumerate(criteria_list):
            key = f"q{index}"
            question = {"type": "choice", "instructions": instructions, "criteria": criteria}
            self._validate_question(state, question)
            proposed = {**pending, key: question}
            if pending and self._size(state, proposed) > 56000:
                results.update(self._rank_questions(state, pending))
                pending = {}
            pending[key] = question
        if pending:
            results.update(self._rank_questions(state, pending))
        return [results[f"q{i}"] for i in range(len(criteria_list))]

    def _size(self, state, questions):
        return len(json.dumps({"model": self.model, "state": state, "questions": questions},
                              ensure_ascii=False).encode())

    def _validate_question(self, state, question):
        if not 1 <= len(question["criteria"]) <= 255:
            raise ValueError("Choice requires between 1 and 255 options")
        if self._size(state, {"question": question}) > 28000:
            raise ValueError("Request is too large; shorten the prompt or conversation.")

    def _rank_questions(self, state, questions):
        for question in questions.values():
            self._validate_question(state, question)
        payload = {"model": self.model, "state": state, "questions": questions}
        for attempt in range(3):
            self.calls += 1
            response = self.http.post("/v1/systemone", json=payload)
            if response.status_code not in (429, 529) or attempt == 2:
                break
            time.sleep(2 ** attempt)
        if response.is_error:
            raise RuntimeError(f"Jev returned HTTP {response.status_code}. Check your key, access, or rate limit.")
        data = response.json()
        results = {}
        for key, question in questions.items():
            probabilities = data["answers"][key]["probabilities"]
            if set(probabilities) != set(question["criteria"]) or not all(
                isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1
                for p in probabilities.values()
            ) or sum(probabilities.values()) <= 0:
                raise ValueError("Jev returned an invalid probability distribution")
            # Live values are rounded; do not require an exact sum of one.
            results[key] = probabilities
        self.input_tokens += data.get("usage", {}).get("input_tokens", 0)
        return results
