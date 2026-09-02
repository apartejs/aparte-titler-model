"""Runs a teacher over a set of messages, via llama-server directly.

llama.cpp directly, not LM Studio: 7x faster at equal quality (measured, see
data/bakeoff/RESULTATS.md). Two processes, one slot each — batching (-np > 1)
collapses the quality.

Requires TITLER_LLAMA_SERVER (path to llama-server.exe) and
TITLER_LLAMA_VENDOR_BIN (its vendor DLL folder) to be set in the environment.

    python eval/run_teacher.py <path.gguf> <label> [--messages f.jsonl] [--processes 2]
"""
import argparse
import io
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LLAMA_SERVER = os.environ.get("TITLER_LLAMA_SERVER")
LLAMA_VENDOR_BIN = os.environ.get("TITLER_LLAMA_VENDOR_BIN")
PORT0 = 8120


def start_server(gguf, port, ngl=99):
    cmd = [LLAMA_SERVER, "-m", gguf, "--host", "127.0.0.1", "--port", str(port),
           "-ngl", str(ngl), "--no-webui", "-np", "1", "-c", "4096"]
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(LLAMA_SERVER) + ";" + LLAMA_VENDOR_BIN + ";" + env.get("PATH", "")
    log = io.open(os.path.join(HERE, "llamacpp_%d.log" % port), "w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log


def wait_ready(port, proc, limit=600):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < limit:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=3)
            return True
        except Exception:
            time.sleep(3)
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
    time.sleep(3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("gguf")
    p.add_argument("label")
    p.add_argument("--messages", default=os.path.join(HERE, "gold_todo.jsonl"))
    p.add_argument("--processes", type=int, default=2)
    p.add_argument("--prompt", default="", help="instructions file")
    p.add_argument("--think", action="store_true",
                   help="let the model reason (no stop sequence, large budget)")
    args = p.parse_args()
    if not LLAMA_SERVER or not LLAMA_VENDOR_BIN:
        raise SystemExit("set TITLER_LLAMA_SERVER (path to llama-server.exe) and "
                          "TITLER_LLAMA_VENDOR_BIN (its vendor DLL folder) in the environment")
    if not os.path.exists(args.gguf):
        raise SystemExit("not found: " + args.gguf)

    print("%s: starting %d servers" % (args.label, args.processes), flush=True)
    servers, specs = [], []
    t0 = time.perf_counter()
    for i in range(args.processes):
        port = PORT0 + i
        servers.append(start_server(args.gguf, port))
        specs.append("t%d@http://127.0.0.1:%d/v1" % (i, port))
    if not all(wait_ready(PORT0 + i, servers[i][0]) for i in range(args.processes)):
        stop_servers(servers)
        raise SystemExit("a server failed to start, see eval/llamacpp_*.log")
    print("  ready in %.0fs" % (time.perf_counter() - t0), flush=True)

    try:
        subprocess.run([sys.executable, "-u", os.path.join(ROOT, "data", "teacher_run.py"),
                        "--messages", args.messages, "--model", ",".join(specs),
                        "--label", args.label, "--no-check"]
                       + ([] if args.think else ["--no-think-api"])
                       + (["--max-tokens", "768"] if args.think else [])
                       + (["--prompt", args.prompt] if args.prompt else []),
                       cwd=ROOT, check=False)
    finally:
        stop_servers(servers)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
