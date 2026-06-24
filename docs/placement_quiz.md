# Placement Quiz — Generative AI, Computational Chemistry & Agentic Engineering

**Purpose:** find your ceiling in each area so we can set per-module starting depth.

**How to take it:** Answer what you can. For anything you don't know, write "skip" or "guess: ..." — partial reasoning is more useful than silence, and a wrong guess tells me more than a blank. **Don't look anything up** — I need your actual ceiling, not your research skills. Where a question says "derive" or "code," a sketch is fine; I'm reading for whether the mental model is there. Answer in any order, in batches across multiple messages if easier.

**Format:** Mix of conceptual and coding/derivation, ordered easy→hard within each strand to locate where you top out. Strands 4 and 5 are pitched a notch higher (you've built a RAG prototype and need the transformer/LLM foundation for the job track); chemistry questions assume your solid background.

**Strand → module mapping:** Strands 1, 2, 6 → Modules 1–3 · Strand 3 → Modules 5–8 · Strand 4 → Modules 4 & 11 · Strand 5 → Modules 12–14.

---

## Strand 1 — Math Foundations

**1.1** Why is the KL divergence $D_{KL}(p \| q)$ asymmetric, and what concretely goes wrong if you swap the arguments when fitting a model $q$ to data $p$? Name the two behaviors (mode-seeking vs. mode-covering) and which direction gives which.

My impression is that if the distribtuions were identical, then there would be no asymmetry. The value would just be 0. When the distribtuions differ, we are essentially computing an ensemble average of the deviation. Switching arguments changes how the distributions deviations are weighted. Switching the arguments would probably lead to objective function collapse/explosion. That is, the objective could be maximized/minimized by collapsing the weighting (which would be learned) distribution. I have no idea regarding mode covering/seeking.

**1.2** The change-of-variables formula for a density under an invertible map includes a $|\det J|$ term. Explain *why* the Jacobian determinant must be there — what would be violated without it?

I didn't know this one. Some brief research suggests that the jacobian accounts for space compression/expansion. I suspect, based on this, that the \int p(x) would not integrate to 1 if this were neglected.

**1.3** State what it means for a function to be E(3)-**invariant** vs. E(3)-**equivariant**, give a physical example of each from molecular systems, and explain why a network built only from pairwise distances is automatically invariant to rotations but throws away information that a tensor-based equivariant network keeps.

E(3) is a Euclidean group, I think it has translations, rotations, and reflections. Maybe more. Invariance means f(g(r)) = f(r), while equivariance means f(g(r)) = g(f(r)) I think. Energies are invariance to global rotations, translations and permutations of identical elements. Forces are equivariant to global rotations. Distances are invariant to global rotations of cartesian axes, so any network based on this will also be invariant. Most of the tensor based networks leverage spherical harmonics as a basis for SO(3) groups, or construct irreps out of them for E(3). This naturally gives the equivariacne property to global rotataions. 

**1.4 (hard)** Spherical harmonics $Y_\ell^m$ are the "irreducible representations of SO(3)" in practice. Explain, as best you can, what that sentence *means* — what does it mean for them to transform within a fixed $\ell$ under rotation, and why does that property make them the natural basis for building equivariant networks? Sketch-level is fine.

Irreducible representations are the fundamental representations of a group. They are the concrete mathematical realizations of abstract members of a group, specifically they act on functions to provide a closed and complete set of all possible transformations within a group. I can't say this precisely, but they also cannot be docomposed into sub components. That is, each transformation g can be represented exactly via a subset of Y_{lm}. 

---

## Strand 2 — Classical ML → Deep Learning

**2.1** You know the bias-variance tradeoff from classical ML. Deep networks often have more parameters than data points yet generalize well. Reconcile this — what does "double descent" claim, and why isn't it a contradiction?

I don't think I'm super familiar with the bias variance tradeoff. I'm thinking that this refers to biased estimators vs potentially flexibility in terms of variance. I have no idea what double descent claims. 

**2.2** Derive backprop for a single linear layer $y = Wx + b$ followed by a scalar loss $L$: given $\partial L/\partial y$, write $\partial L/\partial W$, $\partial L/\partial b$, $\partial L/\partial x$. Dimensions must be consistent.

This is based on the chain rule. $\partial L/\partial W = \partial L/\partial y \cdot x$, $\partial L/\partial b = \partial L/\partial y \cdot 1$, and $\partial L/\partial x = \partial L/\partial y \cdot W$ 

**2.3** Batch norm and layer norm normalize over different axes. State which axis each uses, and explain why layer norm (not batch norm) is the one used in transformers.

Batch norm normalizes over a given subset of the data that is passed to train a model. I don't know what layer normalization is. I'd guess it means that the weights/biases of a given layer are normalized. I'd imagine these are used in transformers because they rely on attention mechanisms which basically use inner products to map correlations and importance which need to be normalized.

**2.4 (hard)** An MLIP predicts energy $E_\theta(\mathbf{r})$ and obtains forces as $\mathbf{F} = -\nabla_{\mathbf{r}} E_\theta$. The training loss includes a force term, so the loss depends on $\nabla_\mathbf{r} E_\theta$. What does this require of the network and the autograd system, and what's the conceptual cost at training time? Why is getting forces *this* way (rather than predicting them as a separate output head) physically important?

It requires the network to be differentiable, and likely have a well defined hessian. There is a substantial cost at training time since the gradients are also trained, whcih requires the hessian for backprop. This is physically important because it makes the resulting force field conservative. 

---

## Strand 3 — Probabilistic & Generative Intuition

**3.1** Write the ELBO and explain each of its two terms (reconstruction and KL-to-prior) in plain language. Then explain what "posterior collapse" is and why the KL term is the culprit.

I found this from the internet. Did not know. In plain english, it is the expected value of the joint distribution minus the entropy of the posterior. I'm guessing posterior collapse would be a trival 0 mean prediction?

**3.2** The reparameterization trick: why can't you backprop through a sample $z \sim \mathcal{N}(\mu, \sigma^2)$ directly, and how does rewriting it as $z = \mu + \sigma \epsilon$, $\epsilon \sim \mathcal{N}(0,1)$ fix it?

It is inherently stocahstic. You'd be backprogataing through a random generator. Reparameterizing allows you to learn mu and sigma (the scaling factor) rather than the stocahstic entreis themselves. So learning the parameters of a distribution rather than the random draws themselves. 

**3.3** In a DDPM, the forward process adds noise over many steps and the model learns to reverse it. Explain why the training objective reduces to predicting the *noise* added at a given step, and what that has to do with score matching (estimating $\nabla_x \log p(x)$).

I have no idea. Not sure what a DDPM is. 

**3.4 (hard)** Score-based SDEs unify DDPM and score matching. There's a "probability-flow ODE" with the *same marginals* as the reverse SDE but no stochasticity. Explain conceptually how a deterministic ODE and a stochastic SDE can produce the same distribution at each time $t$, and one practical thing the ODE form buys you (hint: think likelihood, or fast sampling).

Also no idea.

---

## Strand 4 — Transformer / LLM Mechanics

**4.1** In scaled dot-product attention, $\text{softmax}(QK^\top/\sqrt{d_k})V$ — why the $\sqrt{d_k}$ scaling? What breaks numerically without it?

I'm guessing this distorts the ability to model the outpus as a distribution. Also, it seems like the magnitude of the output would grow with every layer.

**4.2** Self-attention is permutation-equivariant by itself, so transformers need positional information. Contrast sinusoidal absolute encodings with RoPE — what does RoPE do differently, and why does it generalize better to longer sequences?

No idea.

**4.3** Explain the KV-cache: what is cached during autoregressive generation, why it saves computation, and what its memory cost scales with. Why does this make long-context inference expensive?

Again, not sure. This refers to key-value cache, and is obviously part of the keys, queries, values pipeline. I'm assuming this scales quadratically with the size of the query/key. Long context inference would then be expensive due to the cost of the larger inner products. 

**4.4 (hard)** A 1M-token context window exists but "long context ≠ memory" is a common refrain. Give two distinct reasons long context fails to substitute for genuine memory/retrieval in an agent — one about model behavior (attention/effective use), one about systems (cost/latency). Tie this back to why RAG still matters despite huge context windows.

I'm guessing this is because context is more or less pooled and, as mentioned, equivariant. Whereas memory can be recalled on command and as relevant.

---

## Strand 5 — Agentic / LLM-App Tooling

**5.1** Describe the ReAct loop in your own words. What distinguishes an "agent" from a prompt→response chain, mechanically?

No idea.

**5.2** In your RAG prototype: walk me through your chunking and retrieval choices. What was your chunk size/strategy, did you use pure vector search or hybrid, and — critically — how did you know whether retrieval was actually working? (If you didn't measure it, say so; that's a useful signal too.)

We basically arbitrarily set the chunk size. We used vector based semantic similarity search that returned the top k matches. We didn't measure this explicitly, but fielded this out to users and measured if it improved their overall performance and experience. 

**5.3** MCP is described as "USB-C for AI tools." Concretely, what problem does it solve that writing custom function schemas per-model doesn't? What are the moving parts of an MCP server (tools, resources, prompts)?

It can become burdensome to configure a custom API for every potential tool use or desired query. Having a standardized communication and access protocol for models to use overcomes this by standardizing how models interact with each other and tools.

**5.4 (hard)** You're building an agent that can call a tool to run molecular simulations (expensive, slow) and a tool to query a literature database. Design the eval strategy: how do you evaluate this agent's quality offline before production? Address at least: what you'd put in a test set, the problem with LLM-as-judge here, and one agent-specific failure mode (e.g., wrong tool selection, infinite loops, cost blowup) and how you'd catch it. This is the question I most want a real answer to — it's the job's core skill.

I'd test on systems with well defined observables that can also be evaluated cheaply. So the test set would be observables for small systems that could potentially even be evaluated with classical potentials. Maybe a radial distribution function, glass transition temperature, or a free energy from enhanced sampling etc. Using a LLM as a judge is tricky because they can hallucinate. They don't necessarily know what the correct observable is, particularly if we are exploring new systems. So grounding this in the literature or clearly defined physical checks will be critical. There are many possible agent failures. The agent can get pointed in the wrong direction and generate simulations that are effectively meaningless. Maybe it selects the wrong thermostat, or inappropriate parameters, or fails to identify a non-equilibrated simulation. Catching it can be tricky. I'd probably set up an ensemble of sub-agents to challenge each other and escalate to human intervention when there is substantial disagreement.

---

## Strand 6 — Coding & PyTorch

**6.1** What does `loss.backward()` actually do to the computational graph, and why must you call `optimizer.zero_grad()`? What happens if you forget?

It computes the gradients wrt the loss function. The grads must be zerod out or else they can explode. They accumulate otherwise. 

**6.2** What's the difference between `model.eval()` and `torch.no_grad()`? Can you need both at once? Why?

I think eval means we won't need the gradients, we're just doing forward passes. I think no grad means to instantiate a parameter without adding it to the backprop graph.

**6.3** You write a custom loss and get `RuntimeError: element 0 of tensors does not require grad`. What are the likely causes? Name two.

We forgot to wrap the parameter appropriately or it is disonnected from the rest of the computational graph. 

**6.4 (hard)** Sketch (pseudocode or real) a minimal training step for an MLIP where the loss combines energy MSE and force MSE, and forces come from autograd of the energy. The tricky part: you need gradients of the *energy w.r.t. positions* for the forces, AND gradients of the *loss w.r.t. parameters* for the optimizer — a gradient through a gradient. Show how `create_graph=True` (or equivalent) enters.

I don't think i could code this. 
