# geluapprox: fidelity and cost of the three standard GELU forms

GELU is the FFN activation in most transformers, computed three ways: exact (erf), the tanh approximation (gelu_new), and the sigmoid approximation (x*sigmoid(1.702x)). Each f32 form is measured against an f64 exact-erf reference over a dense grid, and timed. Compiled scalar loops at -O3 -march=native (the erf/tanh/exp calls stay scalar, so throughput reflects their cost).

## Accuracy vs the exact-erf reference (max-abs and RMS error over x in [-8, 8])

   variant      max abs err    RMS err
   ----------   -----------    -------
   exact_f32       4.47e-07    6.83e-08
   tanh            4.73e-04    1.74e-04
   sigmoid         2.03e-02    8.40e-03

## Error by x-region (max abs err) - where each approximation is worst

   region      tanh          sigmoid
   ---------   ----------    ----------
   neg_tail      4.12e-04      1.40e-02
   neg_mid       4.73e-04      2.03e-02
   pos_mid       4.73e-04      2.03e-02
   pos_tail      4.12e-04      1.40e-02

Both approximations are worst in the MID regions (|x| ~ 2-3, the curved elbow of GELU) and best in the tails (where GELU is nearly x or nearly 0, easy to match). The error is EXACTLY symmetric between the negative and positive mid regions - the neg_mid and pos_mid maxima agree to f32 rounding. This is provable: GELU(x) - GELU(-x) = x holds for the exact form and for both approximations (Phi(x)+Phi(-x)=1; sigmoid and tanh share the odd-gate identity), so the error is an even function of x. The pre-registered guess that the sigmoid error concentrates on the negative side (P2) therefore could not have held - it is refined to a symmetric mid-elbow.

## Throughput (65536 elements, min over reps)

   variant      ns        relative      elem/ns
   ----------   -------   ----------    -------
   sigmoid        63084       1.00x       1.04
   tanh          271209       4.30x       0.24
   exact_f32     599833       9.51x       0.11

## findings

1. The three forms are NOT interchangeable: the tanh approximation is faithful to exact GELU (max err 4.7e-04) but the sigmoid approximation is ~43x worse (max err 2.0e-02).
2. There is a real speed/accuracy tradeoff: sigmoid is the cheapest (1.0x, one exp) but least accurate; exact-erf is 9.5x slower; the tanh approximation sits in between and is the Pareto sweet spot - a fraction of the exact cost with error two orders of magnitude below sigmoid's.
3. Both approximations' error lives in the mid-range elbow (|x| ~ 2-3), EXACTLY symmetric between positive and negative (the error is a provably even function of x) - not in the tails, and not specific to the negative side as pre-registered.
4. f32 precision is not the issue: the exact form in f32 has error 4.5e-07 against the f64 reference, thousands of times below either approximation - the choice that matters is which approximation, not the float width.

