"""Regenerates the script-alignment fixtures (ADR-052): en_script.wav, ja_script.wav and script_fixtures.json.

Every unit (an English word, a Japanese phrase) is synthesized on its own, its leading and trailing silence is cut at a
fixed level, and the units are joined with fixed gaps, so the start of every unit in the joined file is known exactly:
that is the reference the aligner is measured against. One line of each script is written into the script but never
synthesized, so it is not in the audio (the "not spoken" case).

Needs espeak-ng (English), open-jtalk with hts-voice-nitech-jp-atr503-m001 and the naist-jdic dictionary (Japanese),
and ffmpeg (resampling). Run from anywhere: python tests/fixtures/make_script_fixtures.py
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
RATE = 16000
LEAD = 1.0          # digital silence before the first unit
TAIL = 0.5
WORD_GAP = 0.08     # between units of one line
LINE_GAP = 0.6      # between lines
LEVEL = 0.02        # trim threshold (fraction of full scale)
VOICE = "/usr/share/hts-voice/nitech-jp-atr503-m001/nitech_jp_atr503_m001.htsvoice"
DIC = "/var/lib/mecab/dic/open-jtalk/naist-jdic"

# Each line: (spoken, units). The script is the lines joined with newlines, in this order.
SCRIPTS = {
    "en_script.wav": {
        "language": "en",
        "lines": [
            (True, ["Good", "morning", "and", "welcome", "to", "the", "annual", "meeting."]),
            (False, ["Please", "silence", "your", "phones", "before", "we", "begin."]),
            (True, ["Our", "first", "speaker", "will", "talk", "about", "sleep", "and", "memory."]),
        ],
        "joiner": " ",
        "tts": "espeak-ng en-us, one word at a time",
    },
    "ja_script.wav": {
        "language": "ja",
        "lines": [
            (True, ["本日の", "講演を", "始めます。"]),
            (False, ["携帯電話の", "電源を", "お切りください。"]),
            (True, ["最初の", "演者は", "睡眠と", "記憶について", "話します。"]),
        ],
        "joiner": "",
        "tts": "open-jtalk with hts-voice-nitech-jp-atr503-m001 (CC BY 3.0, Nagoya Institute of Technology), one phrase at a time",
    },
}


def synth(text, language, tmp):
    raw = os.path.join(tmp, "raw.wav")
    out = os.path.join(tmp, "unit.wav")
    if language == "en":
        subprocess.run(["espeak-ng", "-v", "en-us", "-s", "150", "-w", raw, text], check=True)
    else:
        txt = os.path.join(tmp, "unit.txt")
        with open(txt, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        subprocess.run(["open_jtalk", "-x", DIC, "-m", VOICE, "-r", "1.0", "-ow", raw, txt], check=True)
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", raw, "-ac", "1", "-ar", str(RATE),
                    "-c:a", "pcm_s16le", out], check=True)
    with wave.open(out, "rb") as w:
        frames = w.readframes(w.getnframes())
    samples = list(struct.unpack("<%dh" % (len(frames) // 2), frames))
    limit = int(LEVEL * 32767)
    loud = [i for i, v in enumerate(samples) if abs(v) > limit]
    return samples[loud[0]:loud[-1] + 1]


def main():
    truth = {"_comment": ("Script-alignment fixtures (ADR-052), made by make_script_fixtures.py. `units` are the spoken units with "
                          "the exact second their sound starts in the file; the line with spoken=false is in the script, not in the audio."),
             "rate": RATE, "lead": LEAD, "word_gap": WORD_GAP, "line_gap": LINE_GAP}
    with tempfile.TemporaryDirectory() as tmp:
        for name, spec in SCRIPTS.items():
            audio = [0] * int(LEAD * RATE)
            units = []
            first_line = True
            for li, (spoken, words) in enumerate(spec["lines"]):
                if not spoken:
                    continue
                if not first_line:
                    audio += [0] * int(LINE_GAP * RATE)
                first_line = False
                for wi, word in enumerate(words):
                    if wi:
                        audio += [0] * int(WORD_GAP * RATE)
                    pcm = synth(word, spec["language"], tmp)
                    units.append({"line": li, "text": word, "start": round(len(audio) / RATE, 4),
                                  "end": round((len(audio) + len(pcm)) / RATE, 4)})
                    audio += pcm
            audio += [0] * int(TAIL * RATE)
            with wave.open(os.path.join(HERE, name), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(struct.pack("<%dh" % len(audio), *audio))
            lines = [spec["joiner"].join(words) for _spoken, words in spec["lines"]]
            truth[name] = {"language": spec["language"], "script": "\n".join(lines) + "\n",
                           "unspoken_lines": [i for i, (spoken, _w) in enumerate(spec["lines"]) if not spoken],
                           "units": units, "duration": round(len(audio) / RATE, 4), "tts": spec["tts"]}
    with open(os.path.join(HERE, "script_fixtures.json"), "w", encoding="utf-8") as fh:
        json.dump(truth, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
