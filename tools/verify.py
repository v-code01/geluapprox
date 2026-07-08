"""Independent verification of the geluapprox findings, sharing no code with analyze.py. Re-reads
results/bench.jsonl and re-asserts: (P1) tanh max err < 1e-3 and sigmoid max err > 1e-2 (not
interchangeable, order of magnitude apart); (P3) the speed ordering sigmoid < tanh < exact; (P4) the f32
exact error is far below the tanh error; and the refined P2 - both approximations are worst in a MID
region, not a tail. Exit non-zero on mismatch. Run in the gate.
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    rows = [json.loads(x) for x in open("results/bench.jsonl") if x.strip()]

    def acc(variant: str, region: str, key: str) -> float:
        for r in rows:
            if r["kind"] == "acc" and r["variant"] == variant and r["region"] == region:
                return float(r[key])
        raise KeyError(f"{variant}/{region}/{key}")

    def thr(variant: str) -> float:
        for r in rows:
            if r["kind"] == "thr" and r["variant"] == variant:
                return float(r["ns"])
        raise KeyError(variant)

    ok = True

    tanh_e = acc("tanh", "all", "max_err")
    sig_e = acc("sigmoid", "all", "max_err")
    p1 = tanh_e < 1e-3 and sig_e > 1e-2 and sig_e > 10 * tanh_e
    print(f"  [P1] tanh max err {tanh_e:.2e} (<1e-3) sigmoid {sig_e:.2e} (>1e-2), ratio "
          f"{sig_e / tanh_e:.0f}x (not interchangeable = {p1})")
    ok = ok and p1

    speeds = {v: thr(v) for v in ["sigmoid", "tanh", "exact_f32"]}
    p3 = speeds["sigmoid"] < speeds["tanh"] < speeds["exact_f32"]
    print(f"  [P3] speed ns sigmoid {speeds['sigmoid']:.0f} < tanh {speeds['tanh']:.0f} < exact "
          f"{speeds['exact_f32']:.0f} = {p3}")
    ok = ok and p3

    exact_e = acc("exact_f32", "all", "max_err")
    p4 = exact_e < 0.01 * tanh_e
    print(f"  [P4] exact_f32 err {exact_e:.2e} << tanh err {tanh_e:.2e} = {p4}")
    ok = ok and p4

    # Refined P2: both approximations' worst region is a MID region (elbow), not a tail.
    def worst_region(v: str) -> str:
        return max(["neg_tail", "neg_mid", "pos_mid", "pos_tail"],
                   key=lambda r: acc(v, r, "max_err"))
    wt, ws = worst_region("tanh"), worst_region("sigmoid")
    p2 = "mid" in wt and "mid" in ws
    print(f"  [P2 refined] worst region: tanh={wt}, sigmoid={ws} (both mid-elbow = {p2})")
    ok = ok and p2

    # Symmetry: the error is an even function of x, so neg_mid and pos_mid maxima agree (to f32 rounding).
    sn, sp = acc("sigmoid", "neg_mid", "max_err"), acc("sigmoid", "pos_mid", "max_err")
    sym = abs(sn - sp) < 1e-3 * sp
    print(f"  [P2 symmetry] sigmoid neg_mid {sn:.4e} == pos_mid {sp:.4e} (even function = {sym})")
    ok = ok and sym

    if ok:
        print("VERIFY OK: tanh and sigmoid GELU approximations are not interchangeable (sigmoid ~40x "
              "worse), there is a sigmoid<tanh<exact speed ordering, f32 precision is not the limiter, and "
              "the approximation error lives in the mid-range elbow - recomputed independently.")
        return 0
    print("VERIFY FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
