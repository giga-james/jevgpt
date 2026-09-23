from dataclasses import dataclass

from .client import DONE


@dataclass
class Result:
    text: str
    tokens: list[int]
    stop_reason: str


def generate(client, vocabulary, prompt, history=(), max_tokens=128, on_token=None):
    if not prompt.strip():
        raise ValueError("Prompt must not be empty")
    if len(prompt.encode()) > 8000:
        raise ValueError("Prompt must fit within 8,000 UTF-8 bytes")
    if not 1 <= max_tokens <= 512:
        raise ValueError("max_tokens must be between 1 and 512")
    recent = []
    budget = 4000
    for turn in reversed(history):
        size = len(str(turn).encode())
        if size > budget:
            break
        recent.insert(0, turn)
        budget -= size
    tokens = []
    text = ""
    reason = "max_tokens"
    for _ in range(max_tokens):
        shortlist = vocabulary.shortlist(prompt, tokens)
        candidates = {f"t{t}": vocabulary.text[t] for t in shortlist}
        choice, probability = client.choose({
            "conversation": recent,
            "user_message": prompt,
            "assistant_reply_so_far": text,
        }, candidates)
        if choice == DONE:
            reason = "done"
            break
        if choice not in candidates:
            raise ValueError("Jev selected a token outside the shortlist")
        token = int(choice[1:])
        tokens.append(token)
        fragment = vocabulary.text[token]
        text += fragment
        if on_token:
            on_token(fragment, token, probability)
        # Catch six identical tokens or a repeated 2–8 token cycle.
        if any(len(tokens) >= width * 6 and tokens[-width:] * 6 == tokens[-width * 6:]
               for width in range(1, 9)):
            reason = "repetition"
            break
    return Result(text, tokens, reason)
