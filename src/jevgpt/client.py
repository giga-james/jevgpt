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
        payload = {
            "model": self.model,
            "state": state,
            "questions": {"next_token": {
                "type": "choice",
                "instructions": (
                    "Continue the assistant reply to the current user message. Choose the single "
                    "text fragment that should be appended next to assistant_reply_so_far. "
                    "Fragments are literal text, including leading spaces and punctuation. "
                    "Build a helpful, concise, grammatical reply. Do not repeat the user message "
                    "or add role labels. Choose DONE only when the reply is complete."
                ),
                "criteria": criteria,
            }},
        }
        # Conservative local size guard, not an exact count of Jev's tokenizer.
        if len(json.dumps(payload, ensure_ascii=False).encode()) > 28000:
            raise ValueError("Request is too large; shorten the prompt or conversation.")
        for attempt in range(3):
            self.calls += 1
            response = self.http.post("/v1/systemone", json=payload)
            if response.status_code not in (429, 529) or attempt == 2:
                break
            time.sleep(2 ** attempt)
        if response.is_error:
            raise RuntimeError(f"Jev returned HTTP {response.status_code}. Check your key, access, or rate limit.")
        data = response.json()
        answer = data["answers"]["next_token"]
        probabilities = answer["probabilities"]
        if set(probabilities) != set(criteria) or not all(
            isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1
            for p in probabilities.values()
        ) or sum(probabilities.values()) <= 0:
            raise ValueError("Jev returned an invalid probability distribution")
        # Live responses round probabilities (observed totals of 0.99).
        # Argmax is well-defined without assuming the rounded values sum to one.
        choice = max(probabilities, key=probabilities.get)
        self.input_tokens += data.get("usage", {}).get("input_tokens", 0)
        return choice, probabilities[choice]
