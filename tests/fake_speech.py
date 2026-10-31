"""A fake `speech` (speech.cpp) for the speechcpp engine tests: no ggml, no model, no network.

It answers the subcommands and flags the engine uses, in the shapes speech.cpp 0.10.0 writes them (ADR-035;
docs/cli.md, tools/cli/main.cpp, asr.cpp, vad.cpp, tools/common/transcript.cpp, tools/catalog/cache.cpp at
commit 06590811a90041f935a470ba0e2b85346d2fc041):

  speech --version                         speech.cpp 0.10.0, C API 4.0
  speech models --json                     {"version", "directory", "models": [{name, languages, type, files: [...]}], "old"}
  speech devices --json                    {"devices": [{"name", "description", "kind", "memory_total", "memory_free"}]}
  speech vad MODEL.gguf --format json ...  {"file", "regions": [{"start", "end"}]}
  speech asr MODEL.gguf --format json --language L [--timestamps] FILE...
                                           {"file", "text", "stop"[, "segments", "tokens"][, "untimed"]} per file

Like speech.cpp, it keeps models under SPEECH_MODEL_DIR at <owner>--<name>/<sha256>/<file>, and a model is
"fetched" when that file exists. It refuses a model given by name (that would make speech.cpp fetch it), as a
test of "never download at run time". The test drives it through `fake_speech.json` in SPEECH_MODEL_DIR and reads
`calls.jsonl` there (one argv per run). Failures are `speech: <code> (<option>): <message>` on stderr with exit 1.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import wave

# a subset of speech.cpp 0.10.0's tools/catalog/catalog.json: the models the engine knows, and the detector
CATALOG = [
    ("reazonspeech-v2", "sakasegawa/reazonspeech-nemo-v2-GGUF", ["ja"], "recognition", "q8_0",
     [("q8_0", "reazonspeech-nemo-619M-v2-Q8_0.gguf", 660910656), ("f16", "reazonspeech-nemo-619M-v2-F16.gguf", 1240465696),
      ("q4_k", "reazonspeech-nemo-619M-v2-Q4_K.gguf", 357216832)]),
    ("parakeet-tdt_ctc-0.6b-ja", "sakasegawa/parakeet-tdt_ctc-0.6b-ja-GGUF", ["ja"], "recognition", "q8_0",
     [("q8_0", "parakeet-tdt_ctc-0.6B-ja-Q8_0.gguf", 661012416), ("q4_k", "parakeet-tdt_ctc-0.6B-ja-Q4_K.gguf", 357318592)]),
    ("haas-ja-v1.0", "sakasegawa/Haas-ja-v1.0-GGUF", ["ja"], "recognition", "q8_0",
     [("q8_0", "Haas-616M-ja-v1.0-Q8_0.gguf", 656381472)]),
    ("qwen3-asr-0.6b", "sakasegawa/Qwen3-ASR-0.6B-GGUF", ["en", "ja", "zh"], "recognition", "q8_0",
     [("q8_0", "Qwen3-ASR-0.6B-Q8_0.gguf", 841502336), ("q4_k", "Qwen3-ASR-0.6B-Q4_K.gguf", 511168128)]),
    ("qwen3-asr-1.7b", "sakasegawa/Qwen3-ASR-1.7B-GGUF", ["en", "ja", "zh"], "recognition", "q8_0",
     [("q8_0", "Qwen3-ASR-1.7B-Q8_0.gguf", 2176109216)]),
    ("silero-vad", "sakasegawa/silero-vad-GGUF", [], "detection", "f32", [("f32", "silero-vad-309K-v6.2-F32.gguf", 1238624)]),
    ("qwen3-tts-0.6b", "sakasegawa/Qwen3-TTS-12Hz-0.6B-CustomVoice-GGUF", ["en", "ja"], "synthesis", "q8_0",
     [("q8_0", "Qwen3-TTS-12Hz-0.6B-CustomVoice-Q8_0.gguf", 1210000000)]),
]


def sha_of(file: str) -> str:
    """A stand-in for the catalog's SHA-256 of a file (the fake's files are empty)."""
    return hashlib.sha256(file.encode("utf-8")).hexdigest()


def model_dir() -> str:
    d = os.environ.get("SPEECH_MODEL_DIR")
    if not d:
        fail("io", "", "the fake needs SPEECH_MODEL_DIR")
    return d


def path_of(repo: str, file: str) -> str:
    return os.path.join(model_dir(), repo.replace("/", "--"), sha_of(file), file)


def config() -> dict:
    p = os.path.join(model_dir(), "fake_speech.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def fail(code: str, option: str, message: str, status: int = 1) -> None:
    sys.stderr.write(f"speech: {code}{' (' + option + ')' if option else ''}: {message}\n")
    sys.exit(status)


def out(doc) -> None:
    sys.stdout.write(json.dumps(doc, ensure_ascii=False) + "\n")


def known_file(path: str):
    """The catalog row of a model file given by its path; a name or an unknown file is refused."""
    if not path.endswith(".gguf"):
        fail("model_file", "", f"{path}: the fake refuses a model named instead of given as a .gguf path (speech.cpp would fetch it)")
    if not os.path.isfile(path):
        fail("model_file", "", f"cannot open {path}")
    for name, repo, langs, task, _, files in CATALOG:
        for typ, file, _ in files:
            if os.path.abspath(path) == os.path.abspath(path_of(repo, file)):
                return name, langs, task
    fail("model_file", "", f"{path} is not a model file the fake knows")


def flags(args):
    """(positional, {flag: value or True}) the way speech.cpp parses them: a value follows its flag."""
    valued = {"--format", "--language", "--threshold", "--speech-pad-ms", "--min-silence-duration-ms", "--max-speech-duration-s",
              "--min-speech-duration-ms", "--device", "--threads"}
    pos, opts, i = [], {}, 0
    while i < len(args):
        a = args[i]
        if a in valued:
            if i + 1 >= len(args):
                fail("invalid_argument", a, "no value", status=2)
            opts[a] = args[i + 1]
            i += 2
        elif a.startswith("--"):
            if a not in ("--timestamps", "--json"):
                sys.stderr.write(f"speech: there is no flag {a}\n")
                sys.exit(2)
            opts[a] = True
            i += 1
        else:
            pos.append(a)
            i += 1
    return pos, opts


def duration(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def main(argv) -> None:
    cfg = config()
    with open(os.path.join(model_dir(), "calls.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"argv": argv, "cwd": os.getcwd(), "pid": os.getpid(),
                             "env": sorted(k for k in os.environ if k not in ("PWD", "SHLVL", "_"))}) + "\n")
    if not argv:
        sys.exit(2)
    if argv == ["--version"]:
        sys.stdout.write(cfg.get("version_line", "speech.cpp 0.10.0, C API 4.0") + "\n")
        sys.exit(cfg.get("version_exit", 0))
    sub, rest = argv[0], argv[1:]
    if sub in cfg.get("sleep", {}):
        with open(os.path.join(model_dir(), f"sleeping-{sub}.pid"), "w") as fh:
            fh.write(str(os.getpid()))
        time.sleep(float(cfg["sleep"][sub]))
    if sub in cfg.get("fail", {}):
        f = cfg["fail"][sub]
        fail(f["code"], f.get("option", ""), f["message"], f.get("exit", 1))
    if sub == "models" and rest == ["--json"]:
        models = []
        for name, repo, langs, task, typ, files in CATALOG:
            models.append({"name": name, "repository": repo, "revision": "0" * 40, "task": task, "languages": langs, "voice_files": False,
                           "start": [], "type": typ,
                           "files": [{"type": t, "file": f, "size": size, "sha256": sha_of(f), "url": f"https://huggingface.co/{repo}/resolve/x/{f}",
                                      "path": path_of(repo, f), "fetched": os.path.isfile(path_of(repo, f)), "partial": 0} for t, f, size in files]})
        out({"version": "0.10.0", "directory": model_dir(), "models": models, "old": []})
        return
    if sub == "devices" and rest == ["--json"]:
        out({"devices": cfg.get("devices", [{"name": "CPU", "description": "Fake CPU", "kind": "cpu", "memory_total": 1, "memory_free": 1}])})
        return
    if sub == "vad":
        pos, opts = flags(rest)
        name, _, task = known_file(pos[0])
        if task != "detection":
            fail("unsupported", "", f"{name} is not a detection model")
        if opts.get("--format") != "json":
            fail("invalid_argument", "--format", "the fake writes json only")
        want = {"--threshold": "0.5", "--speech-pad-ms": "300", "--min-silence-duration-ms": "500", "--max-speech-duration-s": "10"}
        for k, v in want.items():
            if opts.get(k) != v:
                fail("invalid_argument", k, f"expected {v}, got {opts.get(k)!r}")
        for f in pos[1:]:
            d = duration(f)
            regions = cfg.get("regions")
            if regions is None:
                regions = [[0.0, d]]
            out({"file": f, "regions": [{"start": s, "end": min(e, d)} for s, e in regions]})
        return
    if sub == "asr":
        pos, opts = flags(rest)
        name, langs, task = known_file(pos[0])
        if task != "recognition":
            fail("unsupported", "", f"{name} is a model of speech {task}")
        if opts.get("--format") != "json":
            fail("invalid_argument", "--format", "the fake writes json only")
        lang = opts.get("--language", "auto")
        if lang != "auto" and lang not in langs:
            fail("invalid_argument", "language", f"{name} does not recognize {lang}")
        timed = name.startswith(("reazonspeech", "parakeet", "haas"))
        if opts.get("--timestamps") and not timed:
            fail("unsupported", "timestamps", f"{name} gives no timestamps")
        texts = cfg.get("texts", {})
        status = 0
        for k, f in enumerate(pos[1:]):
            if os.path.isabs(f) or os.sep in f:
                fail("invalid_argument", "", f"the engine passes region files by name in its folder, got {f}")
            d = duration(f)
            text = texts.get(f, "テストです。")
            stop = "model_limit" if f in cfg.get("model_limit", []) else "complete"
            if stop == "model_limit":
                status = 3
            rec = {"file": f, "text": text, "stop": stop}
            if name.startswith("qwen3"):
                rec["languages"] = [lang if lang != "auto" else "ja"]
            if opts.get("--timestamps"):
                if f in cfg.get("untimed", []):
                    rec["untimed"] = "the fake cannot time this file"
                else:
                    rec["segments"], rec["tokens"] = timing(text, d, cfg.get("zero_tokens", False))
            out(rec)
        sys.exit(status)
    sys.stderr.write(f"speech: there is no subcommand {sub}\n")
    sys.exit(2)


def timing(text: str, d: float, zero: bool):
    """Tokens of one character each, spread evenly over [0.1, d - 0.1]; segments cut after 。？！, as FastConformer's are."""
    chars = list(text)
    if not chars:
        return [], []
    a, b = 0.1, max(0.2, d - 0.1)
    step = (b - a) / len(chars)
    tokens = []
    for i, ch in enumerate(chars):
        s = round(a + i * step, 3)
        e = s if (zero and i == 0) else round(a + (i + 1) * step, 3)
        tokens.append({"start": s, "end": e, "text": ch})
    segments, cur = [], []
    for t in tokens:
        cur.append(t)
        if t["text"] in "。？！?!":
            segments.append({"start": cur[0]["start"], "end": cur[-1]["end"], "text": "".join(x["text"] for x in cur)})
            cur = []
    if cur:
        segments.append({"start": cur[0]["start"], "end": cur[-1]["end"], "text": "".join(x["text"] for x in cur)})
    return segments, tokens


if __name__ == "__main__":
    main(sys.argv[1:])
