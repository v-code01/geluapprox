# Pre-registration: geluapprox

Committed to git BEFORE the benchmark was run. Not edited afterward.

## What is measured

The three standard GELU forms, each as an f32 kernel, against an f64 exact-erf reference over x in [-8, 8]:

- **exact**: `x/2 (1 + erf(x/sqrt2))` (the definition; f64 reference and f32 form).
- **tanh** ("gelu_new"): `0.5x (1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))`.
- **sigmoid**: `x * sigmoid(1.702 x)`.

Reported: each approximation's max-abs and RMS error versus the exact reference, the x-region where the
error concentrates, and each form's throughput (compiled scalar loops; the erf/tanh/exp calls stay scalar,
so throughput reflects their relative cost).

## Predictions

**P1 - the approximations are not interchangeable.** The tanh approximation is faithful (max error < 1e-3)
while the sigmoid approximation is far less faithful (max error > 1e-2) - about an order of magnitude
apart. *Falsifier:* both within the same order of magnitude of exact, or tanh worse than sigmoid.

**P2 - the sigmoid error concentrates in the negative-x region.** The sigmoid approximation is worst
around x ~ -2 (where GELU's soft gate shapes small negative activations), and the tanh error worst around
x ~ +2.7. *Falsifier:* the sigmoid error not concentrated in the negative region.

**P3 - a real speed/accuracy tradeoff.** The sigmoid form is the fastest kernel (one sigmoid), the
exact-erf form the slowest (erf), the tanh approximation in between - and the fastest is the least
accurate. *Falsifier:* the speed ordering does not hold, or speed does not trade against accuracy.

**P4 - f32 is not the limiting factor.** The exact form in f32 has error against the f64 reference far
below every approximation's error, so the choice that matters is the approximation, not the precision.
*Falsifier:* the f32 exact error comparable to an approximation's error.

## Commitment

P1 (not interchangeable) and P3 (speed/accuracy tradeoff) are the load-bearing predictions. P2 is a
specific guess about where the sigmoid error lives; whatever the region breakdown shows is reported as-is,
including if the concentration is elsewhere.
