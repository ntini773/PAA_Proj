"""Refresh the Status column of PROGRESS.md from results/.done and results/.fail markers.

    python progress_report.py
"""
import pathlib
import re
import time

ROOT = pathlib.Path(__file__).resolve().parent
DONE, FAIL = ROOT / "results" / ".done", ROOT / "results" / ".fail"


def stamp(p):
    return time.strftime("%m-%d %H:%M", time.localtime(p.stat().st_mtime))


def main():
    md = ROOT / "PROGRESS.md"
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
    print(counts)


if __name__ == "__main__":
    main()
