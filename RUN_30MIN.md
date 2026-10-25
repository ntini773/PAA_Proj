# 30-minute run (server)

Run on the server, from `~/PAA_Proj`. Copy each block as is.

## 0. Before you start (once)

```bash
cd ~/PAA_Proj
mkdir -p results
```
Do this first. Without the `results/` folder, the first `tee` fails (this happened once).
Nothing is lost when it fails: every run also saves its own line in `results/e2e.jsonl`.

## 1. Start the job (about 20 minutes, in the background)

```bash
cd ~/PAA_Proj && mkdir -p results
nohup bash -c '
PY=.pixi/envs/default/bin/python
# 1. main result: bf16 / stock / fused at 256 and 512
for S in 256 512; do for M in bf16 stock fused; do
  $PY scale_e2e.py --size $S --mode $M --steps 14 --reps 2 2>&1 | grep "^{" | tee -a results/e2e.log
done; done
# 2. 1024: stock and fused only
for M in stock fused; do
  $PY scale_e2e.py --size 1024 --mode $M --steps 14 --reps 2 2>&1 | grep "^{" | tee -a results/e2e.log
done
$PY compare_images.py > results/images.log
# 3. CUDA Graph on the real model, 512
for M in stock fused; do
  $PY external_tests_inspired/graph_step_bench.py --size 512 --mode $M 2>&1 | grep "^{" | tee -a results/graph_step.log
done
# 4. where the time goes, 512 only
for M in stock fused; do
  nsys profile -o results/nsys_${M}_512 --force-overwrite=true --trace=cuda,nvtx \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    $PY scale_e2e.py --size 512 --mode $M --steps 2 --reps 1 --profile --tag _nsys --out results/nsys_tmp > results/nsys_${M}_512.log 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true results/nsys_${M}_512.nsys-rep > results/nsys_${M}_512_kern.csv 2>/dev/null
done
$PY classify_nsys.py results/nsys_*_kern.csv > results/nsys_summary.log
$PY make_report.py
echo ALL_DONE
' > run.log 2>&1 &
```

## 2. Watch it

```bash
tail -f run.log            # Ctrl-C stops watching, not the job
cat results/e2e.jsonl      # one line per finished run (always saved)
```
The job is done when the last line of `run.log` is `ALL_DONE`.

## 3. Stop early (if time is short)

```bash
pkill -f scale_e2e.py; pkill -f graph_step_bench.py; pkill -f "nsys profile"
```
Whatever finished is already in `results/`.

## 4. Pack and send back

On the server:
```bash
cd ~/PAA_Proj
tar czf results.tgz results run.log
```
On your laptop:
```bash
scp team13@10.4.25.63:PAA_Proj/results.tgz ~/Documents/projects/PAA/nvfp4/rtx_5000_runs/
tar xzf ~/Documents/projects/PAA/nvfp4/rtx_5000_runs/results.tgz -C ~/Documents/projects/PAA/nvfp4/rtx_5000_runs/
```
Then open `results/REPORT.md`.
If `ALL_DONE` never appeared, run this on the server first: `.pixi/envs/default/bin/python make_report.py`

## 5. What to send me

The contents of `results/e2e.jsonl` and `results/images.log`. I compute the speedup per size.

## If something fails

| Symptom | Fix |
|---|---|
| `results/e2e.log: No such file or directory` | Run `mkdir -p results`. The data is still in `results/e2e.jsonl`. |
| A 1024 run fails | The 1024 transformer is not copied. Skip it. The 256 and 512 results are enough. |
| `nsys: command not found` | Skip block 4. It is optional. |
