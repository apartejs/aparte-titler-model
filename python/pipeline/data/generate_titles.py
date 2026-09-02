"""Generates the corpus's extractive titles with gemma-4-e2b under llama.cpp.

Configuration chosen by the bake-off (data/bakeoff/RESULTATS.md): two
llama-server processes, one slot each. Batching (-np > 1) collapses the
quality, separate processes preserve it.

Resume: ids already present in the output file are skipped, so an
interrupted run can simply be relaunched as-is.

Requires TITLER_LLAMA_SERVER, TITLER_GGUF, and TITLER_LLAMA_VENDOR_BIN to be
set in the environment.

    python -u data/generate_titles.py --limit 200      # smoke test
    python -u data/generate_titles.py                  # the full 50,000

Output: data/canonical/train_titles.jsonl (append, one line per message)
"""
import argparse
import collections
import io
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from teacher_run import call, clean, words, all_words_present  # noqa: E402

LLAMA_SERVER = os.environ.get("TITLER_LLAMA_SERVER")
GGUF = os.environ.get("TITLER_GGUF")
LLAMA_VENDOR_BIN = os.environ.get("TITLER_LLAMA_VENDOR_BIN")
PROMPT = os.path.join(HERE, "prompt_v2.txt")
INPUT = os.path.join(HERE, "canonical", "train_messages.jsonl")
OUTPUT = os.path.join(HERE, "canonical", "train_titles.jsonl")
PORT0 = 8110
MAX_CHARS = 1200

# A refusal is not a title. It is kept, marked as such, rather than thrown
# away: the refusal rate is information about the corpus.
REFUSALS = ("i cannot", "i can't", "i can not", "i'm sorry", "i am sorry", "i'm unable",
            "as an ai", "i apologize")


def start_server(port):
    cmd = [LLAMA_SERVER, "-m", GGUF, "--host", "127.0.0.1", "--port", str(port),
           "-ngl", "99", "--no-webui", "-np", "1", "-c", "4096"]
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(LLAMA_SERVER) + ";" + LLAMA_VENDOR_BIN + ";" + env.get("PATH", "")
    log = io.open(os.path.join(HERE, "canonical", "llamacpp_%d.log" % port), "w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log


def wait_for_server(port, proc, limit=240):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < limit:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=3)
            return True
        except Exception:
            time.sleep(2)
    return False


def stop_servers(servers):
    for proc, log in servers:
        try:
            proc.terminate()
            proc.wait(timeout=30)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            log.close()
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processes", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--output", default=OUTPUT)
    p.add_argument("--input", default=INPUT)
    args = p.parse_args()

    if not LLAMA_SERVER or not GGUF or not LLAMA_VENDOR_BIN:
        raise SystemExit("set TITLER_LLAMA_SERVER, TITLER_GGUF, and "
                          "TITLER_LLAMA_VENDOR_BIN in the environment")
    for f in (LLAMA_SERVER, GGUF, args.input):
        if not os.path.exists(f):
            raise SystemExit("not found: " + f)
    system = io.open(args.prompt, encoding="utf-8").read().strip()

    already = set()
    if os.path.exists(args.output):
        for l in io.open(args.output, encoding="utf-8"):
            try:
                already.add(json.loads(l)["id"])
            except Exception:
                pass
    messages = [json.loads(l) for l in io.open(args.input, encoding="utf-8")]
    if args.limit:
        messages = messages[:args.limit]
    todo_list = [m for m in messages if m["id"] not in already]
    print("%s messages, %s already done, %s to do" % (
        format(len(messages), ","), format(len(already), ","), format(len(todo_list), ",")), flush=True)
    if not todo_list:
        return

    print("starting %d llama.cpp servers" % args.processes, flush=True)
    servers, urls = [], []
    for i in range(args.processes):
        port = PORT0 + i
        servers.append(start_server(port))
        urls.append("http://127.0.0.1:%d/v1" % port)
    if not all(wait_for_server(PORT0 + i, servers[i][0]) for i in range(args.processes)):
        stop_servers(servers)
        raise SystemExit("a server failed to start, see canonical/llamacpp_*.log")
    print("servers ready", flush=True)

    # Same queue as the bake-off: longest to shortest, one thread per server.
    todo = collections.deque(sorted(todo_list, key=lambda m: -m["n_chars"]))
    queue_lock, output_lock = threading.Lock(), threading.Lock()
    output = io.open(args.output, "a", encoding="utf-8")
    count = {"done": 0, "refused": 0, "errors": 0, "compliant": 0}
    t0 = time.perf_counter()

    def worker(url):
        while True:
            with queue_lock:
                if not todo:
                    return
                m = todo.popleft()
            text = m["text"][:MAX_CHARS]
            try:
                raw, secs = call(url, "gemma", system, text, 180, args.max_tokens, True)
                title = clean(raw)
                refused = title.lower().startswith(REFUSALS)
                rec = {"id": m["id"], "title": title, "raw": raw, "seconds": round(secs, 3),
                       "n_words": len(words(title)),
                       "words_present": all_words_present(title, text),
                       "refused": refused}
            except Exception as exc:
                rec = {"id": m["id"], "title": None,
                       "error": ("%s: %s" % (type(exc).__name__, exc))[:200]}
            with output_lock:
                output.write(json.dumps(rec, ensure_ascii=False) + "\n")
                output.flush()
                count["done"] += 1
                count["refused"] += bool(rec.get("refused"))
                count["errors"] += "error" in rec
                count["compliant"] += bool(rec.get("words_present"))
                n = count["done"]
                if n % 500 == 0 or n == len(todo_list):
                    elapsed = time.perf_counter() - t0
                    remaining = (len(todo_list) - n) / (n / elapsed) if n else 0
                    print("  %s/%s  compliant %.0f%%  refused %d  errors %d  |  %.0f titles/h, ~%.0f min left" % (
                        format(n, ","), format(len(todo_list), ","),
                        100 * count["compliant"] / n, count["refused"], count["errors"],
                        n / elapsed * 3600, remaining / 60), flush=True)

    threads = [threading.Thread(target=worker, args=(u,), daemon=True) for u in urls]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        output.close()
        stop_servers(servers)

    duration = time.perf_counter() - t0
    print("\ndone: %s titles in %.1f min (%.0f/h) — compliant %.1f%%, refused %d, errors %d" % (
        format(count["done"], ","), duration / 60, count["done"] / duration * 3600,
        100 * count["compliant"] / max(count["done"], 1), count["refused"], count["errors"]), flush=True)


if __name__ == "__main__":
    main()
