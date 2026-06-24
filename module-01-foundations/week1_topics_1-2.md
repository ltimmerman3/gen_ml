# Week 1 — Information Theory & Probability Foundations

*Phase 1 (foundations). Two topics: KL divergence directionality, then change of variables.*

---

## Topic 1: KL Divergence Directionality

You got the shape of this right in the quiz — zero iff distributions match, swapping reweights the deviations. What was missing was the mechanism and the mode-seeking/mode-covering consequence. That's exactly what we'll close. I'll build it from the definition so the asymmetry becomes obvious rather than memorized.

### The definition, and where the asymmetry lives

$$D_{KL}(p \| q) = \mathbb{E}_{p}\left[\log \frac{p(x)}{q(x)}\right] = \int p(x) \log \frac{p(x)}{q(x)}\, dx$$

Look at what's doing the averaging: the expectation is taken **under $p$**. That single fact is the whole asymmetry. The integrand is weighted by $p(x)$ — so regions where $p$ is large dominate the value, and regions where $p \approx 0$ contribute almost nothing *no matter how wrong $q$ is there*. Swap the arguments and you weight by $q$ instead, and "where it matters to be right" moves entirely.

Your quiz intuition ("changes how deviations are weighted") was pointing right at this. The piece to internalize is which distribution holds the pen.

### Now the consequence — derived, not asserted

Take the fitting problem from your quiz: data $p$ (fixed, possibly multimodal), model $q$ (which you're optimizing, say a single Gaussian that can't cover two modes at once). The two directions force opposite compromises.

**Forward KL, $D_{KL}(p \| q)$ — weighted by the data $p$.** Ask: where does $q$ get punished? Wherever $p(x)$ is large but $q(x)$ is small, the term $p(x)\log\frac{p(x)}{q(x)}$ blows up. As $q(x)\to 0$ at a point where $p(x)>0$, the penalty $\to \infty$. So $q$ is forbidden from being near-zero anywhere the data has mass. With a unimodal $q$ facing bimodal $p$, the only way to put mass on both modes is to stretch across the middle — landing between them, covering everything, including the empty valley. This is **mode-covering** (equivalently mean-seeking). If $q$ is multimodal and $p$ is unimodal, the $q^*$ that minimizes the forward KL will try to cover all of the modes simultaneously, even if the cost is placing mass in regions where $p(x) \approx 0$.

**Reverse KL, $D_{KL}(q \| p)$ — weighted by the model $q$.** Now the penalty is $\int q(x)\log\frac{q(x)}{p(x)}dx$, averaged under $q$. Ask again: where does $q$ get punished? Only where $q$ itself places mass. If $q$ sets $q(x)\approx 0$ somewhere, that region contributes ~nothing to the integral *regardless of $p$* — so $q$ is free to ignore entire modes of $p$ at no cost. What it can't do is place mass where $p$ is near zero (that makes the ratio explode). So $q$ retreats onto a single mode, sits cleanly inside it, and abandons the others. This is **mode-seeking** (equivalently zero-forcing). Where $q$ is low but $p$ is high, the contribution is small; where $q$ is high but $p$ is low, the divergence is large — which is precisely what drives $q$ to hide inside one mode.

A compact way to hold it:

| Direction | Expectation under | Punished for | Result with under-powered $q$ |
|---|---|---|---|
| Forward $D_{KL}(p\|q)$ | data $p$ | being small where $p$ is big | covers all modes, fills valleys (blurry) |
| Reverse $D_{KL}(q\|p)$ | model $q$ | being big where $p$ is small | picks one mode, ignores rest (sharp) |

### Why this matters downstream (the payoff, not a digression)

This isn't a curiosity — it determines the character of generative models you'll build:

- **VAEs** optimize (a bound on) the forward direction. That mode-covering pull is a real reason VAE samples come out blurry — the model is structurally rewarded for hedging across modes rather than committing. You'll feel this concretely in Week 6.
- The forward/reverse split also gives a useful lens on LLM training regimes: **SFT** behaves like minimizing forward KL to the data (mode-covering — spread mass to cover all observed continuations), while **RLHF**-style objectives typically include a *reverse*-KL penalty toward a reference policy, which pulls toward mode-seeking / concentration on high-reward modes. Treat this as a strong heuristic rather than a clean identity — the full RLHF objective isn't simply "minimize reverse KL to the data" — but the same table predicts the qualitative behavior. That's Module 11 territory.

### Canonical references

- **Bishop, *Pattern Recognition and Machine Learning* (2006), §10.1.2** — the standard textbook treatment; the source most papers cite for the mean-seeking/mode-seeking distinction (note Bishop fits with reverse KL in the variational-inference context). Your default reference.
- **Murphy, *Probabilistic Machine Learning: An Introduction* (2022)**, ch. on variational inference — a more modern, very readable version of the same material, freely available as PDF online.
- **Tuan Anh Le's notes** (tuananhle.co.uk/notes/reverse-forward-kl.html) — has the exact bimodal-Gaussian figure and Python to regenerate it; worth running yourself, it makes the asymmetry visceral in a way prose can't.

If you want one thing to do rather than read: code the bimodal-$p$, unimodal-$q$ toy and minimize each direction numerically. Watching $q$ straddle the valley under forward KL and snap onto one mode under reverse KL in ~30 lines is the fastest route to owning this permanently. (It also previews the Week-2 autograd muscle.)

### The gate

Here's a novel case — not one we discussed. Reason it out cold:

> You're training a generative model of molecular conformations. The true distribution $p$ has three well-separated stable conformers (three sharp modes). Your model family $q$ is flexible but, due to a too-small latent capacity, can realistically commit to only one or two of them well.
>
> **(a)** If you train by minimizing forward KL $D_{KL}(p\|q)$, what does the trained $q$ do with the three conformers — and what physically goes wrong with samples you draw from it?
>
> **(b)** If you train by minimizing reverse KL $D_{KL}(q\|p)$ instead, what does $q$ do — and what's the failure mode now?
>
> **(c)** One-liner: which direction would you pick if generating a physically invalid (in-between, non-equilibrium) structure is much worse than missing a conformer — and why?

Answer in your own words. If you reason cleanly to the mode-covering-vs-seeking consequences and connect them to the physical artifacts, you own this.

---

## Topic 2: Change of Variables & Why $|\det J|$

In the quiz you derived the reason this term exists cold — "the integral wouldn't stay normalized to 1." That's the whole conceptual core, so this will be fast: I'm formalizing an instinct you already have, not teaching something new. The payoff is that this formula is the entire mathematical engine of normalizing flows (Module 6), so getting it crisp now pays later.

### The setup

You have a random variable $z$ with a known, simple density $p_z(z)$ (say a standard Gaussian). You push it through an invertible, differentiable map $x = f(z)$ to get a new variable $x$. Question: what's the density $p_x(x)$?

Your instinct says the answer must involve a correction that keeps things normalized. Here's why, and what the correction is.

### The one-line derivation (probability mass is conserved)

The unbreakable fact: probability mass in a region must be preserved by the transformation. The mass $z$ assigns to a tiny box $dz$ must equal the mass $x$ assigns to the image box $dx$:

$$p_x(x)\,|dx| = p_z(z)\,|dz|$$

Rearranging:

$$p_x(x) = p_z(z)\left|\frac{dz}{dx}\right| = p_z(f^{-1}(x))\left|\det \frac{\partial f^{-1}}{\partial x}\right|$$

In multiple dimensions, $\frac{dz}{dx}$ becomes the Jacobian matrix of the inverse map, and $|\cdot|$ becomes $|\det \cdot|$. That's the formula. The $|\det J|$ term is the local volume-rescaling factor — exactly your quiz answer ("space compression/expansion"), now made precise.

### Why the determinant specifically, and why absolute value

Two things worth seeing rather than memorizing:

**Why determinant.** The Jacobian $J$ is the local linear approximation of the map — near a point, $f$ acts like multiplication by $J$. A linear map scales volumes by exactly $|\det J|$ (this is the geometric meaning of the determinant: the volume of the unit cube's image). Density is mass per unit volume; if the map stretches a region's volume by factor $|\det J|$, the density there must drop by the same factor to conserve mass. Hence the determinant, and hence it's the inverse map's Jacobian (or equivalently $1/|\det J_f|$).

**Why absolute value.** Densities must be non-negative. The determinant's sign encodes orientation (whether the map flips space, like a reflection) — that's irrelevant to how much mass sits in a region, so we discard it with $|\cdot|$.

> **One precision point worth holding:** the formula uses the *inverse* map's Jacobian, $|\det \partial f^{-1}/\partial x|$, which equals $1/|\det J_f|$. Both forms appear in the literature — RealNVP and friends are written sometimes one way, sometimes the other — and sign/direction confusion here is a real debugging trap when you build flows in Module 6.

### The payoff (why flows are built around this)

Normalizing flows exploit this formula directly: start with a simple $p_z$ you can sample and evaluate, apply a learned invertible $f$, and you get an *exact* likelihood for $p_x$ — no approximation, unlike the VAE's bound. The entire engineering challenge of flows is then "design expressive invertible maps whose $\det J$ is cheap to compute," because a general $n\times n$ determinant costs $O(n^3)$. That's why architectures like RealNVP use triangular Jacobians (determinant = product of diagonal, $O(n)$). You'll build exactly this in Module 6, and now you'll understand why the architecture is shaped the way it is.

### Canonical references

- **Murphy, *Probabilistic Machine Learning: Advanced Topics* (2023)**, normalizing-flows chapter — the cleanest modern treatment tying the formula directly to flows.
- **Papamakarios et al., "Normalizing Flows for Probabilistic Modeling and Inference" (2021), JMLR** — the definitive survey; §2 derives exactly this. The canonical flows reference, worth bookmarking for Module 6.
- For the pure math, any multivariable calculus text's treatment of the change-of-variables theorem for multiple integrals (the $\int f(x)\,dx = \int f(g(u))|\det Dg|\,du$ you've seen) — the probability version is just this with densities.

### The gate

A novel case, cold:

> Let $z \sim \text{Uniform}(0,1)$ — so $p_z(z) = 1$ on $[0,1]$. Apply the map $x = f(z) = 2z$.
>
> **(a)** Intuitively, before any formula: $x$ now ranges over $[0,2]$. What must $p_x(x)$ be on that interval, and why (use the conservation-of-mass idea)?
>
> **(b)** Now confirm it with the formula — compute $\left|\frac{dz}{dx}\right|$ and check it matches your answer to (a).
>
> **(c)** One sentence: if instead $x = f(z) = z^2$, would $p_x(x)$ be constant over its range or not — and what feature of the map decides that?

Part (c) is the one that tests whether you've got the local nature of the rescaling.
