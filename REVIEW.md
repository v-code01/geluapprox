# Adversarial review: geluapprox

A skeptic's pass over the claims, and why each survives.

## "Everyone knows the GELU approximations differ slightly - this is textbook."
That they differ is known; the useful facts are the magnitudes and the tradeoff. That the sigmoid form is
43x less faithful than the tanh form (a 2% vs 0.05% activation error), that it buys only a ~4x speedup
over tanh and ~9.5x over exact, and that the error is an elbow phenomenon not a tail one - these are
measured numbers an implementer can act on, not folklore. "The approximations differ" is not "sigmoid is
43x worse for a 2x speed win over tanh."

## "Your throughput comparison is unfair - you didn't hand-vectorize erf/tanh/exp."
Correct, and stated as a limitation. The comparison is between the three forms as a compiler produces them
at -O3 -march=native, which is what the vast majority of code actually runs. The throughput therefore
reflects the relative cost of the platform's libm erf/tanh/exp, which is the honest thing to compare when
the question is "which standard form is cheaper." A hand-written polynomial erf would change both its
speed and its accuracy and is explicitly out of scope.

## "The exact form is only 9.5x slower - that's cheap, why approximate at all?"
The study reports exactly this: exact-erf in f32 is essentially reference-accurate (4.5e-7) and only ~9.5x
the cheapest form. If activation throughput is not the bottleneck, exact is a fine choice; the point is
the *sigmoid* approximation trades a large accuracy loss for a speedup that the tanh form mostly matches,
so sigmoid is rarely the right pick. The frontier is the finding, not a recommendation to always
approximate.

## "P2 failed - your prediction was wrong."
P2 predicted the sigmoid error would sit in the negative region; the region breakdown shows it is in the
mid-range elbow and roughly symmetric. That is reported as a refinement, in the README, claims, and this
review, rather than quietly reframed. The load-bearing predictions (P1 not interchangeable, P3 tradeoff)
held; the specific location guess did not, and pre-registration is what makes that visible.

## "f32 vs f64 exact - your reference is not exact."
The f64 exact-erf reference uses the platform's double-precision erf, whose error (~1e-16 relative) is ten
orders of magnitude below the approximation errors being measured (1e-4 to 1e-2). It is an exact oracle
for this purpose, and all f32 forms are measured against the same reference.

## "verify.py just echoes analyze.py."
verify.py re-reads results/bench.jsonl and recomputes the fidelity ordering, the speed ordering, the
f32-sufficiency ratio, and the worst-region argmax with its own logic, sharing no code with analyze.py.

## Pre-registration honesty
All four predictions were committed before the benchmark; P1/P3/P4 held and P2 is reported as refined.
PREREG.md is unedited and flags P2 up front as the specific guess that could miss.
