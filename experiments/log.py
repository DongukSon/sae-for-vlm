"""Make piped/redirected output readable as a log file.

- Progress bars redrawn with '\r' (tqdm, wget) are collapsed to their final state,
  which shows the total elapsed time, e.g. "100%|██| 37/37 [00:20<00:00, 1.8it/s]".
  Bars that get erased (tqdm leave=False) are dropped, as they are on a terminal.
- Every line gets a "[YYYY-mm-dd HH:MM:SS]" prefix; blank lines and ANSI escapes are dropped.

Usage: some_command 2>&1 | python -u experiments/log.py
"""
import codecs
import os
import re
import signal
import sys
import time

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
SEP_RE = re.compile(r"(\r\n|\r|\n)")


def emit(line):
    line = ANSI_RE.sub("", line).rstrip()
    if line.strip():
        sys.stdout.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {line}\n")
        sys.stdout.flush()


def main():
    # Ctrl-C reaches the whole pipeline: keep running until EOF so the last lines are not lost
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Incremental, so a multi-byte character split across two reads is not mangled
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    text = ""     # regular output being accumulated up to the next '\n'
    bar = None     # latest complete state of the progress bar being redrawn
    bar_cur = ""   # bar state currently being written (after the last '\r')
    in_bar = False

    def settle_bar():
        # A blank redraw erases the bar: tqdm.write() does it before printing (and redraws the
        # bar afterwards), leave=False does it on close. Either way that state is not a final one
        nonlocal bar, bar_cur
        if bar_cur.strip():
            bar = bar_cur
        elif bar_cur:
            bar = None
        bar_cur = ""

    while True:
        chunk = os.read(0, 65536)
        for piece in SEP_RE.split(decoder.decode(chunk, final=not chunk)):
            if piece == "\r":
                if in_bar:
                    settle_bar()
                elif text.strip():
                    bar = text  # e.g. "Downloading...\r" is also a redrawn status line
                text = ""
                in_bar = True
            elif piece in ("\n", "\r\n"):
                if in_bar:
                    settle_bar()
                    in_bar = False
                    text = ""
                if text.strip():
                    emit(text)
                elif bar is not None:
                    emit(bar)  # the '\n' that closes a finished bar
                    bar = None
                text = ""
            elif in_bar:
                bar_cur += piece
            else:
                text += piece
        # Each bar redraw is a single write: text arriving in a later write without a
        # leading '\r' (e.g. a print() while the bar is running) is a new line, not part of the bar
        if in_bar:
            settle_bar()
            in_bar = False
        if not chunk:
            break

    emit(text)
    if bar is not None:
        emit(bar)


if __name__ == "__main__":
    main()
