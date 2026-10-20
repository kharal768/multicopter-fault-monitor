"""Run one method of twin_baseline, one domain at a time (keeps memory low)."""
import sys, os, glob, gc
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twin_baseline as tb
from rflymad import load_case, case_table
src, out, name, mode = sys.argv[1:5]
parts = []
# per-domain partial results are saved, so an interrupted run resumes at the next unfinished domain
part_dir = os.path.join(out, ".parts", name)
os.makedirs(part_dir, exist_ok=True)
for digit in os.environ.get("DOMAINS", "321"):                                   # Real, HIL, SIL
    part = os.path.join(part_dir, f"{digit}.csv")
    if os.path.exists(part):
        parts.append(pd.read_csv(part))
        print("domain", digit, "loaded", flush=True)
        continue
    files = sorted(glob.glob(os.path.join(src, f"Case_{digit}*.csv")))
    if not files:
        continue
    data = pd.concat([load_case(f) for f in files], ignore_index=True)
    r, _ = tb.evaluate(data, mode)
    d = case_table(data).merge(r, on="case_id")
    d.to_csv(part, index=False)
    parts.append(d)
    del data, r; gc.collect()
    print("domain", digit, "done", flush=True)
pd.concat(parts, ignore_index=True).to_csv(os.path.join(out, f"{name}_results.csv"), index=False)
print("done", name)
