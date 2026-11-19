#!/usr/bin/env bash
# One command to confirm that the eval2 steps will work, before spending GPU hours. About 10-15 minutes on the server.
#   bash smoke_eval2.sh            then:  bash run_eval2.sh
# Each line prints PASS / FAIL / WARN. FAIL = fix before running run_eval2.sh (the failing log is shown). WARN = an optional
# piece is missing (Sage attention, mslk, CUDA kernels) and the matching variants will be skipped or fail on their own.
cd "$(dirname "$0")"
PY=${PY:-python}
# SKIP_CUDA=1: skip the CUDA-extension checks (e.g. a laptop with nvcc < 12.8). SKIP_MODELS=1: skip the new-model smoke.
have_nvcc() { [ -z "$SKIP_CUDA" ] && command -v nvcc > /dev/null; }
mkdir -p logs/smoke results/.done
bad=0
ok()   { echo "PASS  $1"; }
warn() { echo "WARN  $1"; }
fail() { echo "FAIL  $1"; bad=1; [ -n "$2" ] && tail -n 12 "$2" | sed 's/^/        /'; }
step() { # step "name" logfile command...
  local name=$1 log=$2; shift 2
  if "$@" > "$log" 2>&1; then ok "$name"; else fail "$name" "$log"; fi
}

echo "== environment"
$PY - > logs/smoke/env.log 2>&1 <<'EOF' && ok "python imports (torch, diffusers, torchao, transformers, lpips, sentencepiece, matplotlib)" || fail "python imports" logs/smoke/env.log
import torch, diffusers, torchao, transformers, lpips, sentencepiece, matplotlib
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
assert torch.cuda.is_available()
EOF
for tool in nvcc nsys ncu; do command -v $tool > /dev/null && ok "$tool found" || warn "$tool not found (needed for CUDA kernels / nsys / ncu stages)"; done
$PY -c "import sageattention" 2> /dev/null && ok "sageattention (Sage2) installed" || warn "sageattention not installed: bf16+sage2 variants skipped (pip install sageattention --no-build-isolation)"
$PY -c "import sageattn3" 2> /dev/null && ok "sageattn3 (Sage FP4) installed" || warn "sageattn3 not installed: sage3 variants skipped"
$PY -c "import mslk" 2> /dev/null && ok "mslk installed (stock_tk usable)" || warn "mslk not installed: stock_tk configs will fail (pip install --pre mslk)"
[ -f prompts/embeds_pixart.pt ] && ok "prompts/embeds_pixart.pt (eval prompts)" || fail "prompts/embeds_pixart.pt missing: run 'bash run_all.sh embeds'"
[ -f prompts/embeds_calib.pt ] && ok "prompts/embeds_calib.pt (calibration prompts)" || warn "prompts/embeds_calib.pt missing: stage E01 creates it"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/      GPU: /'

echo "== CPU checks (tiny model, 71 checks)"
step "tests_cpu" logs/smoke/cpu.log bash tests_cpu/run_cpu_checks.sh
grep -E "FAIL" logs/smoke/cpu.log | head -5

echo "== GPU checks on real PixArt weights, 256 px"
step "tests_gpu (layer bit-exactness, compile engagement, w4a16, DiTFastAttn, limited-interval CFG)" logs/smoke/gpu.log bash tests_gpu/run_gpu_checks.sh 256
grep -E "FAIL|mean rel L2|rel L2 vs|plan counts" logs/smoke/gpu.log | head -8

echo "== CUDA kernel (needs nvcc 13): static-scale quantize against Triton"
if have_nvcc; then
  step "CUDA swizzled quantize kernel builds and matches Triton" logs/smoke/cuda.log $PY bench/cuda_vs_triton.py --only cuda_swizzled
else warn "skipped (no nvcc or SKIP_CUDA=1)"; fi

echo "== every new config runs one step (256 px, 3 iterations)"
source profiling/configs.sh
for cfg in bf16_compile module_triton module_triton_compile module_noshare_compile module_selective_compile module_cuda_compile module_best_xcudnn bf16_trim_compile; do
  case $cfg in
    module_cuda_compile|module_best_xcudnn) have_nvcc || { warn "$cfg skipped (needs nvcc for the CUDA kernel)"; continue; };;
  esac
  [ "$cfg" = module_best_xcudnn ] && [ ! -f results/static_scales_256.json ] && [ ! -f results/static_scales.json ] && { warn "$cfg skipped (needs static scales: run stage E04)"; continue; }
  step "step_target $cfg" logs/smoke/step_$cfg.log $PY step_target.py --size 256 $(cfg_args $cfg) --config smoke_$cfg --iters 3 --warmup 2
done

echo "== timing tools"
step "profile_stages (per-stage table)" logs/smoke/stages.log $PY timing/profile_stages.py --variant module+be=triton --size 256 --iters 2 --out smoke_stage_timing.jsonl
step "traffic ledger" logs/smoke/ledger.log $PY analysis/traffic_ledger.py --size 512

echo "== quality tool on 2 prompts (random-tiny CLIP: checks the code path only)"
step "run_quality (stock, module)" logs/smoke/quality.log $PY quality/run_quality.py --size 256 --max-prompts 2 --steps 4 --clip random-tiny \
  --out results/smoke_quality --variants stock module+be=triton

echo "== new models (forward in bf16 / stock / fused at their smallest size)"
for m in sana_600m sd35_medium flux_schnell; do
  [ -n "$SKIP_MODELS" ] && { warn "model_smoke $m skipped (SKIP_MODELS=1)"; continue; }
  step "model_smoke $m" logs/smoke/model_$m.log $PY model_smoke.py --model $m --size 1024
done

echo
if [ $bad -eq 0 ]; then echo "SMOKE: no FAIL. Next: bash run_eval2.sh   (bash run_eval2.sh --list shows the stages)";
else echo "SMOKE: FAILURES above. Fix them or send logs/smoke/*.log before run_eval2.sh"; fi
exit $bad
