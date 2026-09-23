import argparse
import os
from pathlib import Path
import sys
import time

from dotenv import load_dotenv
import httpx

from .chat import generate
from .client import JevClient
from .vocabulary import Vocabulary
from .hierarchy import Hierarchy


def main():
    parser = argparse.ArgumentParser(description="A chatbot made of Jev classification decisions")
    parser.add_argument("prompt", nargs="?", help="Omit for interactive chat; /quit exits, /reset clears history")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selection", choices=("shortlist", "hierarchical"), default="hierarchical",
                        help="Token selection strategy (default: hierarchical)")
    parser.add_argument("--corpus", type=Path, help="UTF-8 text to use instead of the tiny bundled corpus")
    parser.add_argument("--trace", action=argparse.BooleanOptionalAction, default=False,
                        help="Show routing decisions and token probabilities on stderr (default: disabled)")
    parser.add_argument("--dry-run", action="store_true", help="Print initial choices without calling Jev")
    args = parser.parse_args()
    if args.prompt is not None and not args.prompt.strip():
        parser.error("Prompt must not be empty")
    load_dotenv(Path.cwd() / ".env")
    if not 1 <= args.max_tokens <= 512:
        parser.error("--max-tokens must be between 1 and 512")
    key = os.getenv("TYPESAFE_API_KEY", "").strip()
    if not key and not args.dry_run:
        parser.error("Set TYPESAFE_API_KEY in .env or your environment")
    client = None
    try:
        vocab = Vocabulary(args.corpus.read_text() if args.corpus else None, args.seed)
        hierarchy = Hierarchy(vocab) if args.selection == "hierarchical" else None
        if args.dry_run:
            if hierarchy is not None:
                import json
                print(json.dumps(hierarchy.options(hierarchy.root), ensure_ascii=False, indent=2))
                return
            for token in vocab.shortlist(args.prompt or "Hello!", []):
                print(f"t{token}\t{vocab.text[token]!r}")
            print("DONE\tEnd the reply")
            return
        client = JevClient(key, os.getenv("JEV_MODEL", "jev-1.13.0"))
        history = []
        while True:
            prompt = args.prompt if args.prompt is not None else input("You: ")
            if args.prompt is None and prompt.strip() == "/quit":
                break
            if args.prompt is None and prompt.strip() == "/reset":
                history.clear()
                continue
            if not prompt.strip():
                continue
            if args.prompt is None:
                print("Jev: ", end="", flush=True)
            start = time.monotonic()
            calls, usage = client.calls, client.input_tokens

            def emit(fragment, token, probability):
                print(fragment, end="", flush=True)
                if args.trace:
                    print(f"\n[t{token} p={probability:.4f} text={fragment!r}]", file=sys.stderr)

            def decision(depth, choice, probability, count):
                if args.trace:
                    print(f"\n[depth={depth} choice={choice} local_p={probability:.4f} options={count}]",
                          file=sys.stderr)

            result = generate(client, vocab, prompt, history, args.max_tokens, emit,
                              hierarchy=hierarchy, on_decision=decision)
            print()
            if args.trace:
                print(f"[{result.stop_reason}; {len(result.tokens)} tokens; "
                      f"{client.calls - calls} calls; {client.input_tokens - usage} input tokens; "
                      f"{time.monotonic() - start:.1f}s]", file=sys.stderr)
            history.append({"user": prompt, "assistant": result.text})
            if args.prompt is not None:
                break
    except (EOFError, KeyboardInterrupt):
        print()
    except (OSError, ValueError, RuntimeError, KeyError, httpx.HTTPError) as error:
        # Avoid logging requests or headers that contain credentials.
        print(f"Error: {error if not isinstance(error, httpx.HTTPError) else 'Network request failed'}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        if client:
            client.close()
