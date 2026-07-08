# geluapprox: the three standard GELU forms are not interchangeable

GELU is the FFN activation in most transformers, and it ships in three forms: the exact definition
`x/2 (1 + erf(x/sqrt2))`, the tanh approximation `gelu_new = 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))`
(GPT-2 and many others), and the sigmoid approximation `x * sigmoid(1.702 x)`. They are often treated as
the same function. This measures how faithful each is to exact GELU (against an f64 erf reference), where
the error lives, and how fast each is - the speed/accuracy frontier an implementer actually chooses on.
Standalone C++.

## Pre-registration

Four predictions were committed to git (`PREREG.md`) before benchmarking: (P1) the approximations are not
interchangeable (tanh faithful, sigmoid an order of magnitude worse); (P2) the sigmoid error concentrates
in the negative-x region; (P3) a real speed/accuracy tradeoff (sigmoid fastest, exact slowest); (P4) f32
is not the limiter. **P1, P3, P4 held. P2 was refined** - the error concentrates in the mid-range elbow
(|x| ~ 2-3) exactly symmetrically (the error is a provably even function of x), not the negative side.

## Results

**Accuracy** vs the f64 exact-erf reference, over x in [-8, 8]:

```
 variant       max abs err    RMS err
 exact_f32       4.47e-07     6.83e-08
 tanh            4.73e-04     1.74e-04
 sigmoid         2.03e-02     8.40e-03
```

**Throughput** (65536 elements, min over reps):

```
 variant       ns        relative      elem/ns
 sigmoid       63084       1.00x         1.04
 tanh         271209       4.30x         0.24
 exact_f32    599833       9.51x         0.11
```

1. **The three forms are not interchangeable.** The tanh approximation is faithful to exact GELU (max
   error 4.7e-4), but the sigmoid approximation is **43x worse** (max error 2.0e-2). Swapping
   `x*sigmoid(1.702x)` for exact GELU introduces a 2% activation error; swapping the tanh form introduces
   0.05%.

2. **A real speed/accuracy tradeoff, with tanh as the sweet spot.** The sigmoid form is the cheapest (one
   exp), the exact-erf form ~9.5x slower (erf is expensive), and the tanh approximation sits in between at
   ~4.3x the sigmoid cost. So the fastest form is by far the least accurate, and the tanh approximation is
   the Pareto choice - a fraction of the exact cost with error two orders of magnitude below sigmoid's.

3. **The error lives in the mid-range elbow.** Both approximations are worst at |x| ~ 2-3 (the curved part
   of GELU) and near-exact in the tails (where GELU is ~x or ~0), EXACTLY symmetric between positive and
   negative - the error is a provably even function of x (GELU(x)-GELU(-x)=x holds for all three forms), so
   P2's negative-side guess could not have held.

4. **f32 precision is not the issue.** The exact form in f32 has error 4.5e-7 against the f64 reference -
   thousands of times below either approximation. The choice that matters is *which approximation*, not
   the float width.

## The one-line finding

The tanh and sigmoid GELU approximations are not interchangeable: the sigmoid form is ~10x faster than
exact but 43x less faithful than the tanh form (a 2% vs 0.05% activation error, worst in the |x|~2-3
elbow), so the tanh approximation is the speed/accuracy sweet spot and f32 precision is never the limiter.

## Reproduce

```
./reproduce.sh 400        # build, benchmark (self-contained), analyze, verify
./scripts/gate.sh         # C++ -Werror build + test, ruff, mypy, ASCII, leak scan, independent verify
```

The benchmark generates its own inputs, so it needs no model or data. `tools/verify.py` recomputes the
fidelity ordering, the speed ordering, the f32-sufficiency, and the worst-region result from
`results/bench.jsonl` with its own arithmetic, sharing no code with `analyze.py`. The C++ test asserts the
fidelity ordering (tanh close, sigmoid far) and that each kernel matches its scalar definition.

## Limitations and falsifiers

- One ISA (ARM), one compiler at `-O3 -march=native`, the standard approximation constants. Absolute ns
  are machine-specific and the erf/tanh/exp costs are the platform's libm; the *ordering* and the fidelity
  ratios are the claims. A hand-vectorized polynomial erf/tanh/exp would change the absolute throughput
  (and its own accuracy), which this study does not explore - it measures the standard compiled forms.
- The error magnitudes are activation-space errors, not a downstream model-quality claim; whether a 2%
  sigmoid-approx error matters depends on the model and is not measured here.
- **Falsifier (refined):** P2 predicted the sigmoid error would concentrate in the negative region; the
  error is a provably even function of x (exactly symmetric mid-elbow), so that specific prediction could
  not have held and is reported as refined, not confirmed.

MIT licensed. The oracle is an f64 exact-erf GELU (`std::erf`); error is exact max-abs/RMS deviation;
latency is minimum-over-reps monotonic-clock time. No LLM judgement.
