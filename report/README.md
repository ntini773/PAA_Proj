# eval2 report

`eval2_report.pdf` (19 pages): what was done, why, what it implies, with diagrams, worked examples, real images, and consistency checks.

Rebuild from server results (the `results_eval2.tgz` you rsync back):

```bash
mkdir -p /tmp/srv && tar xzf results_eval2.tgz -C /tmp/srv           # gives /tmp/srv/results and /tmp/srv/logs
# first-run E11/E12 logs (the reruns overwrote them): from the results-eval2 branch, commit f35fab8
mkdir -p /tmp/hist && git show f35fab8:latest/logs/eval2/E11_512.log  > /tmp/hist/E11_512_first.log
git show f35fab8:latest/logs/eval2/E11_1024.log > /tmp/hist/E11_1024_first.log
git show f35fab8:latest/logs/eval2/E12.log      > /tmp/hist/E12_first.log
python report/make_facts.py        --data /tmp/srv --hist /tmp/hist     # one facts.json: every number comes from here
python report/check_consistency.py --data /tmp/srv                      # 13 'no two truths' checks
python report/build_report.py      --data /tmp/srv                      # figures, text, PDF (headless Chrome)
```

Files: `make_facts.py` (read results, fixed selection rules), `common.py`, `make_figures.py`, `traffic.py`, `check_consistency.py`, `build_report.py`.
Needs the pixi env (torch, torchao, matplotlib, Pillow), `ncu` (to re-read the saved `.ncu-rep` files) and `google-chrome`.
