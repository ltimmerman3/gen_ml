# Log — Module 01, Topic: KL Divergence Directionality

**Date:** Week 1
**Week / type:** Week 1 (Theory)

## What this topic was
KL divergence is asymmetric because the expectation is taken under one distribution. Forward KL D(p‖q) is weighted by data p → mode-covering (q spreads to cover all modes, fills valleys, "blurry"). Reverse KL D(q‖p) is weighted by model q → mode-seeking / zero-forcing (q snaps onto one mode, ignores others, stays valid).

## The gate
**Question:** 3-conformer target p, under-capacity q. Predict behavior under forward vs reverse KL and the physical artifact; pick a direction when invalid in-between structures are costlier than missing a conformer.
**My answer (cold):** (a) forward blurs the three conformers → samples are non-physical in-between geometries. (b) reverse snaps to one well → valid but incomplete catalog. (c) reverse KL, because zero-forcing keeps mass off the barriers so every sample is a real conformer.
**Verdict:** passed clean.

## What I got wrong or fuzzy
- Originally had the *mechanism* of asymmetry wrong — thought it was about objective collapse, not about which distribution does the weighting.
- Did not know mode-seeking vs mode-covering at all going in.
- Build bug: truncated grid [-5,5] with modes at ±3 silently broke reverse KL (q spread mass past the grid edge, so the zero-forcing penalty went uncounted). Forward was unaffected because it's weighted by p, which had no mass out there. Fixed by widening the grid.

## Key thing to retain
**expectation-under-which-distribution → what it's punished for → mode behavior.** Everything regenerates from that. Corollary: the two directions differ in *where they're numerically sensitive*, not just what they prefer — which is why a too-small grid broke reverse but not forward.
Sanity reflex: check ∫density ≈ 1 over the grid *at the optimizer's solution*, not just at the start.

## Canonical references
- Bishop, *Pattern Recognition and Machine Learning* (2006), §10.1.2 — standard source for mean/mode-seeking.
- Murphy, *Probabilistic ML: An Introduction* (2022), variational inference chapter — modern, free PDF.
- Tuan Anh Le's notes (tuananhle.co.uk/notes/reverse-forward-kl.html) — the bimodal-Gaussian figure + regenerating code.

## Artifacts
- module-01-foundations/kl_directionality.ipynb (forward = straddle, reverse = snap-to-mode)
