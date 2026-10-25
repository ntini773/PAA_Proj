# Next commands (plain version)

Run on the server. Each command saves its output to a log file.

## 0. Update the code and check nobody else is using the GPU
```bash
cd ~/PAA_Proj && git pull
nvidia-smi
```
If another user's program is listed with a lot of memory, the timings will be too slow. Run anyway if you are short on time, and note it.

## 1. The test (about 6 minutes)
```bash
bash run_next.sh
```
It saves everything to `results/next_<time>.log` and prints it too.

What it does, in plain words:
- **A.** Lets our fast kernel also handle the biggest layer it skipped (the second layer of the MLP, 4608 numbers wide), then runs the model at 1024, 512 and 256 pixels. It checks the pictures stay the same as the stock version and shows how long a step takes now.
- **B.** Prints the step times from the earlier run so you can compare.
- **C.** Shows how far each number format (fp16, fp8, NVFP4) drifts from bf16.
- **D.** Prints three old results we never looked at: the picture check, the CUDA Graph test, and why `torch.compile` failed.

## 2. Send me
The whole log file: `cat results/next_*.log`

## 3. If something fails
Copy the last 20 lines of the error.
