"""Bake-off: which teacher respects the extractive constraint.

One loaded instance = one dedicated thread. Every thread draws from a shared
queue, so an instance never has two requests in flight and none sits idle
while there is still work left.

    lms load google/gemma-4-e2b -c 4096 --identifier g2-1 -y
    python -u data/teacher_run.py --model g2-1,g2-2 --label gemma-4-e2b --limit 15

Output: data/bakeoff/results/<label>__<variant>.jsonl
"""
import argparse
import collections
import io
import json
import os
import re
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MESSAGES = os.path.join(HERE, "bakeoff", "messages.jsonl")
RESULTS = os.path.join(HERE, "bakeoff", "results")
WORD = re.compile(r"[a-z0-9]+")


def words(text):
    return WORD.findall(text.lower())


def is_subsequence(title, message):
    """Do the title's words appear in the message, in order?"""
    tw, mw = words(title), words(message)
    if not tw:
        return False
    it = iter(mw)
    return all(w in it for w in tw)


def all_words_present(title, message):
    tw, mw = words(title), set(words(message))
    return bool(tw) and all(w in mw for w in tw)


THINK = re.compile(r"<think>.*?</think>", re.S)


def clean(raw):
    """One line, with no quotes or trailing punctuation.

    Some models (Qwen3) render their reasoning right inside content, between
    tags, instead of in a separate field: it is stripped out first.
    """
    text = THINK.sub("", raw or "")
    # An opening tag that's never closed: the reasoning was truncated, and
    # nothing usable is left after it. Without this we'd mistake gibberish
    # for a valid title.
    if "<think>" in text:
        text = text.split("<think>")[0]
    lines = text.strip().splitlines()
    title = lines[0].strip() if lines else ""
    return title.strip("\"“”«»‘’' ").rstrip(".!?:;").strip()


def call(base_url, model, system, message, timeout, max_tokens,
         no_think_api=False, attempts=3):
    """One title. Any reasoning arrives in a separate field.

    Generation is not cut short: if the model reasons, it needs the room to
    finish before writing the title, otherwise `content` comes back empty.
    """
    body_dict = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": message}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if no_think_api:
        # llama-server accepts this field (Lemonade doesn't). Without it,
        # gemma-4 goes back into reasoning mode and burns the 512-token
        # budget per title.
        body_dict["chat_template_kwargs"] = {"enable_thinking": False}
        body_dict["stop"] = [chr(10)]
    body = json.dumps(body_dict).encode("utf-8")
    last = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(
                base_url.rstrip("/") + "/chat/completions", data=body,
                headers={"Content-Type": "application/json"})
            started = time.perf_counter()
            data = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            secs = time.perf_counter() - started
            msg = data["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            if not content and (msg.get("reasoning_content") or "").strip():
                raise RuntimeError("empty content: the model reasoned without "
                                   "concluding, increase --max-tokens")
            return content, secs
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise last


def loaded_models(base_url):
    """What LM Studio has in memory, with the reserved context."""
    root = base_url.rstrip("/").rsplit("/v1", 1)[0]
    try:
        raw = urllib.request.urlopen(root + "/api/v0/models", timeout=15).read()
    except Exception:
        return None
    return {m["id"]: m.get("loaded_context_length")
            for m in json.loads(raw).get("data", []) if m.get("state") == "loaded"}


def check_loaded(args):
    """Never let a model get loaded without the user knowing.

    Calling the API with a model that isn't loaded makes LM Studio load it
    with its default config, often at maximum context, which throws
    everything off.
    """
    if args.no_check:
        print("model check skipped (--no-check)")
        return
    loaded = loaded_models(args.base_url)
    if loaded is None:
        raise SystemExit("LM Studio unreachable at " + args.base_url)
    if not loaded:
        raise SystemExit("no model loaded in LM Studio")
    print("loaded: " + ", ".join(f"{k} (ctx={v})" for k, v in loaded.items()))
    missing = [n for n, _ in args.models if n not in loaded]
    if missing:
        raise SystemExit("not loaded: " + ", ".join(missing))
    unexpected = [k for k in loaded if k not in {n for n, _ in args.models}]
    if unexpected:
        print("WARNING: other resident models: " + ", ".join(unexpected))
    for name, _ in args.models:
        if (loaded.get(name) or 0) > 16384:
            print(f"WARNING: {name} at context {loaded[name]}, reload at 4096")


def run_variant(args, variant, messages):
    path_in = args.prompt or os.path.join(HERE, "prompt_%s.txt" % variant)
    system = io.open(path_in, encoding="utf-8").read().strip()
    if args.no_think:
        # Qwen3 switch. In the SYSTEM prompt, never in the message: otherwise
        # "/no_think" counts as a word present in the message and throws off
        # the extractive-compliance measurement.
        system += chr(10) + "/no_think"

    # Two-ended queue, sorted from longest to shortest.
    # Fast instances draw from the long end, slow ones (--slow, typically a
    # CPU instance) from the short end. Without this, an instance 6x slower
    # grabs a big message at startup and single-handedly bounds the total
    # time: measured at 37.2s of wall time for a run whose GPU work fit in 26.4s.
    todo = collections.deque(
        sorted(enumerate(messages), key=lambda kv: -kv[1]["n_chars"]))
    todo_lock = threading.Lock()

    def take(slow):
        with todo_lock:
            if not todo:
                return None
            return todo.pop() if slow else todo.popleft()
    results = [None] * len(messages)
    done = {"n": 0}
    lock = threading.Lock()

    def worker(instance, url):
        """Tied to a single instance: it never has two requests in flight."""
        slow = instance in args.slow
        while True:
            item = take(slow)
            if item is None:
                return
            index, msg = item
            text = msg["text"][:args.max_chars]
            try:
                raw, secs = call(url, instance, system, text,
                                 args.timeout, args.max_tokens, args.no_think_api)
                title = clean(raw)
                record = dict(msg, variant=variant, instance=instance, raw=raw,
                              title=title, seconds=round(secs, 3),
                              n_words=len(words(title)),
                              subsequence=is_subsequence(title, msg["text"]),
                              words_present=all_words_present(title, msg["text"]))
            except Exception as exc:
                record = dict(msg, variant=variant, instance=instance, title=None,
                              error=("%s: %s" % (type(exc).__name__, exc))[:200])
            results[index] = record
            with lock:
                done["n"] += 1
                print("  ... %d/%d" % (done["n"], len(messages)))

    threads = [threading.Thread(target=worker, args=(name, url), daemon=True)
               for name, url in args.models
               for _ in range(args.per_instance)]
    started = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - started

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "%s__%s.jsonl" % (args.label.replace("/", "_"), variant))
    with io.open(path, "w", encoding="utf-8") as fh:
        for record in results:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    ok = [r for r in results if r.get("title")]
    total = len(results)
    print("\n--- %s [%s]  %d instances x %d ---"
          % (args.label, variant, len(args.models), args.per_instance))
    print("  succeeded        : %d/%d" % (len(ok), total))
    if ok:
        lat = sorted(r["seconds"] for r in ok)
        print("  median latency   : %.2fs   (wall: %.1fs)" % (lat[len(lat) // 2], wall))
        good = sum(1 for r in ok if 3 <= r["n_words"] <= 5)
        print("  3-5 word length  : %d/%d (%.0f%%)" % (good, len(ok), good / len(ok) * 100))
        if variant == "extractive":
            present = sum(1 for r in ok if r["words_present"])
            subseq = sum(1 for r in ok if r["subsequence"])
            print("  words from msg   : %d/%d (%.0f%%)" % (present, len(ok), present / len(ok) * 100))
            print("  order respected  : %d/%d (%.0f%%)" % (subseq, len(ok), subseq / len(ok) * 100))
        by_instance = {}
        for r in ok:
            by_instance[r["instance"]] = by_instance.get(r["instance"], 0) + 1
        print("  breakdown        : " + ", ".join("%s=%d" % kv for kv in sorted(by_instance.items())))
    bad = [r for r in results if r.get("error")]
    if bad:
        print("  failures         : %d   ex: %s" % (len(bad), bad[0]["error"][:110]))
    print("  written to       : " + os.path.relpath(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="comma-separated identifiers, one per loaded instance")
    parser.add_argument("--label", default="", help="output file name")
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--prompt", default="", help="instructions file")
    parser.add_argument("--no-think-api", action="store_true",
                        help="chat_template_kwargs.enable_thinking=false + stop (llama-server)")
    parser.add_argument("--no-think", action="store_true",
                        help="append /no_think to the system prompt (Qwen3)")
    parser.add_argument("--no-check", action="store_true",
                        help="skip the model check (non-LM-Studio backend)")
    parser.add_argument("--slow", default="",
                        help="slow instances (CPU): they take the short messages")
    parser.add_argument("--per-instance", type=int, default=1,
                        help="simultaneous requests per instance")
    parser.add_argument("--variant", choices=["free", "extractive", "both"],
                        default="extractive")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--messages", default=MESSAGES,
                        help="jsonl of messages (default: the 50 from the bake-off)")
    args = parser.parse_args()
    # "name" uses --base-url; "name@http://host:port/v1" has its own server,
    # which makes it possible to mix several processes (or several backends).
    args.models = []
    for raw in args.model.split(","):
        raw = raw.strip()
        if not raw:
            continue
        name, _, url = raw.partition("@")
        args.models.append((name, url or args.base_url))
    args.label = args.label or args.models[0][0]
    args.slow = {m.strip() for m in args.slow.split(",") if m.strip()}

    check_loaded(args)
    messages = [json.loads(line) for line in io.open(args.messages, encoding="utf-8")]
    if args.limit:
        messages = messages[:args.limit]
    variants = ["free", "extractive"] if args.variant == "both" else [args.variant]
    for variant in variants:
        run_variant(args, variant, messages)


if __name__ == "__main__":
    main()
