#!/usr/bin/env bash
# Regenerates the speech fixtures. Needs: open-jtalk + hts-voice-nitech-jp-atr503-m001 (Japanese),
# espeak-ng (English), ffmpeg. The committed WAV/MP4 files are the outputs of this script; the
# reference texts live in fixtures.json. Speech starts after LEAD seconds of digital silence.
set -euo pipefail
cd "$(dirname "$0")"
LEAD=1.0
TAIL=0.5
JA_TEXT="本日の講演を始めます。よろしくお願いします。まず最初に、会場の音響設備についてご説明します。"
EN_TEXT="Good morning everyone. Welcome to the conference. Let's begin the first session with a short introduction."
VOICE=/usr/share/hts-voice/nitech-jp-atr503-m001/nitech_jp_atr503_m001.htsvoice
DIC=/var/lib/mecab/dic/open-jtalk/naist-jdic
tmp=$(mktemp -d)
printf '%s\n' "$JA_TEXT" > "$tmp/ja.txt"
open_jtalk -x "$DIC" -m "$VOICE" -r 1.0 -ow "$tmp/ja_raw.wav" "$tmp/ja.txt"
espeak-ng -v en-us -s 150 -w "$tmp/en_raw.wav" "$EN_TEXT"
pad() { # pad <in> <out>: LEAD s silence + speech + TAIL s silence, mono 16 kHz PCM
  ffmpeg -y -hide_banner -loglevel error -i "$1" -af "adelay=${LEAD}s:all=1,apad=pad_dur=${TAIL}" -ac 1 -ar 16000 -c:a pcm_s16le "$2"
}
pad "$tmp/ja_raw.wav" ja_short.wav
pad "$tmp/en_raw.wav" en_short.wav
# video with audio: Japanese, 1.5 s gap, English; tiny H.264 picture so the file stays small
ffmpeg -y -hide_banner -loglevel error -i ja_short.wav -i en_short.wav \
  -filter_complex "[0:a]apad=pad_dur=1.0[a0];[a0][1:a]concat=n=2:v=0:a=1[a]" -map "[a]" -ac 1 -ar 16000 -c:a pcm_s16le "$tmp/lecture.wav"
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$tmp/lecture.wav")
ffmpeg -y -hide_banner -loglevel error -f lavfi -i "testsrc2=size=320x180:rate=15" -i "$tmp/lecture.wav" -t "$dur" \
  -c:v libx264 -preset veryfast -crf 30 -pix_fmt yuv420p -c:a aac -b:a 48k -shortest lecture_short.mp4
rm -rf "$tmp"
ls -la ja_short.wav en_short.wav lecture_short.mp4
