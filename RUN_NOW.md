# Run now: copy models, then benchmark

Two machines:
- **LOCAL** = your laptop (has the downloaded models)
- **SERVER** = `team13@10.4.25.63` (runs the benchmarks)

Server: `team13@10.4.25.63`. Server home: `/data/cs3406/team13`.
Model cache (assumed, confirm in Step 1): `/data/cs3406/team13/.cache/huggingface`.

---

## Step 1. SERVER: find the cache folder

```bash
ssh team13@10.4.25.63 'echo "HF_HOME=${HF_HOME:-unset}"; mkdir -p ~/.cache/huggingface/hub; ls -d ~/.cache/huggingface/hub'
```
- `HF_HOME=unset` (expected): the cache is `/data/cs3406/team13/.cache/huggingface`. Use the commands below as written.
- `HF_HOME` set to another path: replace `/data/cs3406/team13/.cache/huggingface` with that path everywhere below, and export `HF_HOME` in Step 4.

## Step 2. LOCAL: copy the models

Why two copies: the real model files live in `hub/blobs/`. The model folders hold only
shortcuts (symlinks) to them. You must copy both.

```bash
cd ~/.cache/huggingface/hub
DST=team13@10.4.25.63:/data/cs3406/team13/.cache/huggingface/hub/

# 2a. the shortcuts (small)
rsync -avP --exclude '*.incomplete' models--PixArt-alpha--* $DST

# 2b. the real files (about 7 GB)
find models--PixArt-alpha--* -path '*/blobs/*' -type l -exec readlink {} \; \
  | sed 's|^\.\./\.\./||' | sort -u > ~/pixart_blobs.txt
rsync -avP --files-from=$HOME/pixart_blobs.txt ./ $DST
```
If the connection drops, run the same command again. It resumes.

## Step 3. SERVER: check the copy worked

```bash
ls -laL ~/.cache/huggingface/hub/models--PixArt-alpha--*/snapshots/*/transformer/
```
Pass: three lines show `diffusion_pytorch_model.safetensors` with size **2443492488**.
Fail: size 0 or "No such file". Repeat Step 2b.

## Step 4. SERVER: quick test (2 minutes)

```bash
cd ~/PAA_Proj
# only if Step 1 showed a custom HF_HOME:  export HF_HOME=<that path>
pixi run smoke
```
Pass: one line of JSON with `"fused_calls"` above 0.
It is normal for `fallback_calls` to be in the hundreds.
Fail: paste the last 20 lines.

## Step 5. SERVER: run the benchmarks

```bash
cd ~/PAA_Proj
nohup bash -c '
  bash run_all.sh e2e
  bash run_all.sh nsys
  bash external_tests_inspired/run_inspired.sh
  bash run_all.sh stretch
  pixi run python make_report.py
' > run.log 2>&1 &
tail -f run.log          # Ctrl-C stops watching, not the job
```

What each line does, and how long it takes:

| Line | Question it answers | Time |
|---|---|---|
| `run_all.sh e2e` | How much faster is the whole model, at 256, 512, 1024px? Are the images identical? | ~30 min |
| `run_all.sh nsys` | Where does the time go: matrix math, attention, or our small kernel? | ~20 min |
| `run_inspired.sh` | Is the gain from fusion or from less launch cost? | ~15 min |
| `run_all.sh stretch` | Batch 4 and `torch.compile`. Optional. | ~15 min |

If time runs out, keep the first line (`e2e`). It is the main result.

Look at progress:
```bash
cat results/e2e.log        # one JSON line per finished run
```

## Step 6. SERVER: pack the results

```bash
cd ~/PAA_Proj
tar czf results.tgz results RUNS_LOG.md RESULTS_server_partial.md
```

## Step 7. LOCAL: download the results

```bash
scp team13@10.4.25.63:PAA_Proj/results.tgz ~/Documents/projects/PAA/nvfp4/rtx_5000_runs/
```
Then open `results/REPORT.md`.

---

## If something fails

| Message | Meaning | Fix |
|---|---|---|
| `does not appear to have a file named diffusion_pytorch_model.safetensors` | The real files are missing (Step 2b not done or wrong path). | Redo Step 2b, then Step 3. |
| `OfflineModeIsEnabled` | Same cause. The run is offline on purpose. | Same fix. |
| Size 0 or no file in Step 3 | Wrong path in `DST`, or the copy was cut off. | Repeat Step 2 with the right path. |
| Fails only at 1024 | The 1024 transformer is not copied. | Redo Step 2b. Or remove `1024` from the `for S in ...` lines in `run_all.sh`. |
