import json
from collections import Counter
from types import SimpleNamespace

import httpx

from jevgpt.client import JevClient
from jevgpt.hierarchy import Hierarchy


def test_hierarchy_batches_independent_branches_into_four_requests():
    widths = []

    def respond(request):
        payload = json.loads(request.content)
        widths.append(len(payload["questions"]))
        answers = {}
        for key, question in reversed(list(payload["questions"].items())):
            options = question["criteria"]
            answers[key] = {"probabilities": {k: 1 / len(options) for k in options}}
        return httpx.Response(200, json={"answers": answers, "usage": {"input_tokens": 10}})

    vocab = SimpleNamespace(text={i: chr(97+i) for i in range(16)}, frequency=Counter())
    tree = Hierarchy(vocab, fanout=4, leaf_size=2, beam_width=3, per_leaf=1)
    client = JevClient("fake", transport=httpx.MockTransport(respond))
    try:
        choice, _ = tree.choose(client, {"user_message": "Hi"})
        assert choice.startswith("t")
        assert widths == [1, 3, 3, 1]
        assert client.calls == 4
        assert client.input_tokens == 40
    finally:
        client.close()


def test_large_batches_split_and_keep_result_order():
    sizes = []

    def respond(request):
        data = json.loads(request.content)
        sizes.append(len(request.content))
        answers = {k: {"probabilities": {c: 1.0 for c in q["criteria"]}}
                   for k, q in data["questions"].items()}
        return httpx.Response(200, json={"answers": answers})

    client = JevClient("fake", transport=httpx.MockTransport(respond))
    try:
        result = client.rank_many({}, [{str(i): "x" * 20000} for i in range(3)], "Rank")
        assert result == [{str(i): 1.0} for i in range(3)]
        assert client.calls == 2
        assert all(size <= 56000 for size in sizes)
    finally:
        client.close()
