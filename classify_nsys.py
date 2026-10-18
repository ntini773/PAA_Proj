"""Bucket nsys cuda_gpu_kern_sum CSVs into gemm / attention / our fused quant /
other elementwise+reduce (= unfused norm-quant debris) and print % of GPU time.
    python classify_nsys.py results/nsys_*_kern.csv
"""
import sys, csv, re, io, collections
B = [("fused_quant", r"_fused_"),
     ("attention", r"flash|fmha|attention|softmax|mem_eff|sdpa"),
     ("gemm", r"gemm|cutlass|nvjet|cublas|scaled_mm|sm1\d\d_xmma|splitK"),
     ("elementwise/reduce/copy", r".*")]
for f in sys.argv[1:]:
    txt = open(f).read(); i = txt.find("Time (%)")
    rows = list(csv.DictReader(io.StringIO(txt[i:])))
    tot = collections.Counter(); names = collections.defaultdict(list); T = 0
    for r in rows:
        t = float(r["Total Time (ns)"]); n = r["Name"]; T += t
        for b, rx in B:
            if re.search(rx, n, re.I):
                tot[b] += t; names[b].append((t, n[:70])); break
    print(f"\n== {f}  total kernel time {T/1e6:.1f} ms")
    for b, _ in B:
        print(f"   {b:26s} {100*tot[b]/T:5.1f}%  {tot[b]/1e6:9.2f} ms")
    print("   top other kernels:", [n for _, n in sorted(names['elementwise/reduce/copy'], reverse=True)[:4]])
    print("   top gemm kernels :", [n for _, n in sorted(names['gemm'], reverse=True)[:2]])
