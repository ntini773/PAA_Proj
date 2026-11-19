"""Refresh the Status column of the trackers from results/.done and results/.fail markers.

    python progress_report.py

Trackers: PROGRESS.md (first experiment round: N, K, S, P, R, Q, T, V, M rows) and PROGRESS_EVAL2.md (eval2: E rows).
A row is any table line starting with `| <Letter><2 digits> |`; its last cell becomes `todo`, `DONE <time>` or `FAILED <time>`.
Both files are rewritten in place; a counts marker `<!-- counts -->...<!-- /counts -->` is kept up to date.
"""
import pathlib
import re
import time

ROOT = pathlib.Path(__file__).resolve().parent
DONE, FAIL = ROOT / "results" / ".done", ROOT / "results" / ".fail"
TRACKERS = ["PROGRESS.md", "PROGRESS_EVAL2.md"]


def stamp(p):
    return time.strftime("%m-%d %H:%M", time.localtime(p.stat().st_mtime))


def update(md):
    out, counts = [], dict(done=0, failed=0, todo=0)
    for line in md.read_text().splitlines():
        m = re.match(r"^\| ([A-Z]\d{2}) \|", line)
        if m:
            tid = m.group(1)
            cells = line.rstrip().rstrip("|").split("|")
            if (FAIL / tid).exists():
                status, k = f"FAILED {stamp(FAIL / tid)}", "failed"
            elif (DONE / tid).exists():
                status, k = f"DONE {stamp(DONE / tid)}", "done"
            else:
                status, k = "todo", "todo"
            counts[k] += 1
            cells[-1] = f" {status} "
            line = "|".join(cells) + "|"
        out.append(line)
    text = "\n".join(out) + "\n"
    text = re.sub(r"<!-- counts -->.*?<!-- /counts -->",
                  f"<!-- counts -->{counts['done']} done, {counts['failed']} failed, {counts['todo']} todo<!-- /counts -->", text)
    md.write_text(text)
    return counts


def main():
    for name in TRACKERS:
        md = ROOT / name
        if md.exists():
            print(name, update(md))


if __name__ == "__main__":
    main()
