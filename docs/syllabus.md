# Generative AI, Computational Chemistry & Agentic Engineering
### Exhaustive Syllabus — Canonical Version

**Legend:** ★ job-critical spine (path to an agentic-AI offer) · ◆ chemistry application · ○ depth/enrichment, take à la carte · ☁ cloud/CI-CD touchpoint (full track detailed at the end).

**Three parts:** I — Theoretical Spine (ground-up foundation) · II — Chemistry Application (de-emphasized but covered) · III — LLMs & Agentic Engineering (the job-converting core). Roughly 16 modules, 10–14 weeks at exhaustive depth. Part III can run in parallel from ~week 3. The ★ path alone is a complete, employable route through the bloat; everything else is there when a topic grabs you. A **Cloud & CI/CD track** (GCP) runs alongside the whole syllabus, acquiring each infra skill exactly when an artifact justifies it — see the dedicated section at the end.

---

## PART I — Theoretical Spine

### Module 0 — Foundations & Framing
- ★ Generative vs. discriminative vs. supervised-regression framings.
- ★ The generative model taxonomy (autoregressive, latent-variable, flows, EBMs, GANs, diffusion) and what each makes tractable — likelihood, sampling, or latent structure (no model gets all three free).
- ◆ Where LLMs and MLIPs each sit in this map.

### Module 1 — Mathematical Bedrock
- ★ Tensor calculus; Jacobians, Hessians; the chain rule as the engine of backprop.
- ★ Probability: distributions, expectation, change of variables, pushforward measures.
- ★ Information theory: entropy, cross-entropy, KL divergence and its asymmetry, mutual information, the ELBO as a KL decomposition, Jensen's inequality.
- ★ Optimization: SGD, momentum, Adam, AdamW; loss-landscape geometry; why overparameterization helps.
- ○ Measure-theoretic probability: densities vs. measures; when a density is well-defined.
- ○ Estimation: MLE↔KL equivalence, score matching, Fisher information.
- ○ Stochastic processes: Markov chains, Brownian motion, SDEs, Fokker–Planck (feeds diffusion).
- ○◆ Group & representation theory: groups, irreps, tensor products, spherical harmonics, Wigner matrices — built here so Module 6's equivariance has foundations. Understanding, not implementation.
- **Practice:** ★ implement reverse-mode autodiff + an MLP from scratch in NumPy; ○ verify the ELBO bound and KL/entropy estimators numerically.

### Module 2 — Deep Learning Core
- ★ Backpropagation derived by hand; autograd internals.
- ★ MLPs; initialization (Xavier/He); normalization (batch vs. layer, and why transformers use layer norm); residual connections and why they enable depth.
- ★ Regression vs. classification objectives (MSE/MAE, cross-entropy); uncertainty-aware/heteroscedastic losses.
- ★ Embedding layers: the bridge from discrete tokens to vectors.
- CNNs: convolution, pooling, receptive fields (lighter — motivational for spatial inductive bias).
- RNNs/LSTMs: sequence modeling and the gradient pathologies that motivate attention.
- ○ Regularization, bias-variance in the overparameterized regime, double descent.
- ○◆ The gradient-of-a-scalar idea (energy→forces) introduced conceptually.
- **Practice:** ★ port the NumPy autograd engine to PyTorch and confirm gradients match; build reusable layers.
- ☁ **Cloud (foundation):** everything in Git/GitHub from here on; stand up GitHub Actions running lint + pytest on push. Trivial now, but establishes the CI reflex and a clean commit history for the capstone repo.

### Module 3 — Geometric Deep Learning
*(Lighter — understand and reason about equivariant architectures; not a from-scratch build.)*
- ★/○ Graph topology; message passing (MPNNs); graph convolutions and attention.
- ◆ Invariance vs. equivariance made concrete: scalar energy is E(3)-invariant, forces are E(3)-equivariant; why distance-only nets are invariant but discard directional info.
- ◆ The tensor-product / spherical-harmonic construction (e3nn) — conceptually.
- **Practice:** ○ build a small invariant and a small equivariant layer; verify symmetry properties numerically.

### Module 4 — Sequence Modeling & the Transformer ★
*(The bridge into the LLM track — do it thoroughly.)*
- ★ Autoregressive factorization p(x)=∏ p(xᵢ | x₍<ᵢ₎).
- ★ Self-attention from scratch: Q/K/V, scaled dot-product (and why √dₖ), multi-head.
- ★ Positional encodings: sinusoidal vs. RoPE and length generalization.
- ★ Full architectures: encoder–decoder vs. decoder-only; the transformer block end to end.
- ★ Tokenization (BPE); sampling (temperature, top-k, nucleus).
- ○ Architectural variants: GQA/MQA, mixture-of-experts, state-space models (Mamba).
- **Practice:** ★ implement a decoder-only GPT from scratch and train it on a small corpus (ground-up, non-negotiable for the job).

### Module 5 — Latent Variable Models & VAEs
- ★ Autoencoders → VAEs; the ELBO (reconstruction + KL-to-prior).
- ★ The reparameterization trick; posterior collapse and why the KL term causes it.
- ◆ Discrete latents: VQ-VAE (important for molecular work).
- β-VAE and the disentanglement/reconstruction tradeoff.
- **Practice:** ◆ train a VAE to encode/generate valid molecules via SMILES or 2D graphs; visualize and traverse the latent space.

### Module 6 — Normalizing Flows
- Exact likelihood via the change-of-variables formula (where Module 1's pushforward measures pay off).
- Coupling layers (RealNVP); autoregressive flows (MAF/IAF); the expressivity–tractability tradeoff in the Jacobian.
- Continuous normalizing flows as the bridge to diffusion.
- **Practice:** implement RealNVP on a 2D toy density; verify exact log-likelihood.

### Module 7 — Energy-Based Models & GANs
- EBMs: unnormalized densities, the partition function problem, contrastive divergence, Langevin sampling.
- GANs: the minimax game, mode collapse, Wasserstein GANs and the theory behind them.
- Where these sit relative to likelihood-based methods.

### Module 8 — Diffusion & Score-Based Models ★
- ★ Forward/reverse diffusion; why the objective reduces to noise prediction.
- ★ DDPM end to end; the variational view.
- Score-based modeling; denoising score matching; the ∇ₓ log p(x) connection.
- Score-SDEs (Song et al.): unifying DDPM and score matching; the probability-flow ODE and what it buys (likelihood, fast sampling).
- ★ Guidance: classifier and classifier-free; conditional generation; latent diffusion.
- ○ Sampling accelerators: DDIM, distillation, consistency models.
- **Practice:** ★ implement DDPM from scratch on images; ○ confirm the score-SDE view coincides with DDPM.
- ☁ **Cloud (training):** this is the first genuinely GPU-hungry build — move off any laptop/Colab scratchpad onto real provisioned compute. Vertex AI Workbench for controlled-GPU notebooks, or a Compute Engine spot/preemptible GPU instance for cheaper raw time (and more real cloud mechanics: SSH, env setup, the things that break). Set billing alerts before you start.

---

## PART II — Chemistry Application

### Module 9 — MLIPs, Conceptually ◆
*(De-emphasized on implementation — understand and use, don't rebuild MACE.)*
- ◆ The lineage as ideas: SchNet → DimeNet → PaiNN → NequIP → MACE, and what each buys geometrically.
- ◆ Energy+forces(+stress) training; conservation (a true force field is the negative energy gradient); foundation potentials (MACE-MP) and the universal-potential trend.
- ○ The gradient-through-a-gradient training mechanic (create_graph=True).
- **Practice:** ◆ use a pretrained MACE-MP as a callable tool and probe its limits; ○ (optional, deeper) implement a SchNet-style model on rMD17, verifying forces equal autograd energy gradients.

### Module 10 — Generative Chemistry ◆
- ◆ Equivariant diffusion for 3D molecule/conformer generation (reusing Module 3's concepts).
- ◆ Representations: SMILES/SELFIES vs. graph vs. 3D point clouds.
- ◆ Property-conditioned generation; validity/uniqueness/novelty evaluation.
- **Practice:** ◆ fine-tune a chemical language model (SELFIES) **or** build a transformer for retrosynthetic pathway prediction; adapt/run an existing molecular generative model for 3D conformers rather than building from scratch.

---

## PART III — LLMs & Agentic Engineering (the job-converting core)

### Module 11 — LLMs in Depth & Fine-Tuning ★
- ★ From Module 4's transformer to modern LLMs: pretraining vs. post-training, instruction tuning, RLHF and DPO, scaling laws.
- ★ Context windows and their limits; why long context ≠ memory.
- ★ Inference internals: KV-cache, quantization, batching, serving (vLLM-style).
- ★ Hands-on fine-tuning: LoRA/QLoRA end to end; the fine-tune-vs-prompt-vs-retrieve decision.
- ◆ RL for property optimization: steering generative chemistry toward target properties.
- ○ Distillation, speculative decoding, structured/constrained decoding.
- **Practice:** ★ fine-tune an open model with LoRA on a domain task; ◆ steer a molecular generator toward a property objective.

### Module 12 — Context Engineering & RAG ★
*(The backbone of most enterprise agents — heavily screened-for, go deep.)*
- ★ Embeddings; chunking strategies; vector DBs (pgvector/Pinecone/Weaviate); hybrid search; re-ranking.
- ★ Evaluation of retrieval quality — honestly measured, not vibes.
- **Practice:** ★ build a RAG system over a chemistry corpus (papers/datasets) and measure retrieval quality.
- ☁ **Cloud (data):** Cloud Storage for corpora/artifacts; run the vector DB as **pgvector on Cloud SQL** (Postgres) rather than a managed vector service — teaches managed databases and connection security, both transferable infra knowledge.

### Module 13 — Agents: Tool Use & Orchestration ★
- ★ The ReAct loop (reason→act→observe) as the core primitive; what mechanically distinguishes an agent from a prompt→response chain.
- ★ Tool/function-calling design: schemas, least-privilege tools, kill switches (OWASP LLM Top 10 awareness).
- ★ MCP: build an MCP server (concrete, repeatedly-given advice); tools/resources/prompts; why it beats per-model custom schemas.
- ★ Orchestration: single-agent-with-tools vs. multi-agent (Level 2–3 is the production sweet spot; multi-agent is often demo-ware); LangGraph/CrewAI/OpenAI Agents SDK as patterns, not one library.
- ○ Memory architectures; A2A protocols; planning beyond ReAct.
- **Practice:** ★◆ build an agent that calls real tools — a pretrained MLIP, RDKit, a literature DB — via MCP.
- ☁ **Cloud (CI stage 2):** Dockerize the agent and its MCP server; have GitHub Actions build the image on push and push it to **Artifact Registry**. This is continuous *integration* of real artifacts — the bridge between trivial-CI and real deployment.

### Module 14 — Evaluation & Observability ★
*(The single biggest hiring signal.)*
- ★ Eval design for prompts, RAG, and agents; building offline test suites.
- ★ LLM-as-judge and its pitfalls; ensemble/agreement gates; when judges are untrustworthy.
- ★ Observability: tracing, monitoring, online vs. offline eval; feedback loops from production back into evals.
- ◆ Chemistry-specific evaluation: validity, uniqueness, novelty, synthetic accessibility (SA) scoring.
- **Practice:** ★ build an eval harness for the Module 13 agent (portfolio gold); ◆ evaluate a generative-chemistry pipeline on novelty + SA.

### Module 15 — Production: Cost, Reliability, Deployment ★
*(The "2 a.m. under load" competence that separates mid from senior.)*
- ★ Cost optimization: caching, routing, model selection (the 40–70% lever and a lab-vs-production screen).
- ★ Latency and streaming; token-level ops.
- ★ Agent failure modes under load: wrong tool selection, infinite loops, cost blowup — and how to catch each.
- ★ Deployment: containerization (Docker), serving, CI for prompts/agents.
- **Practice:** ★ deploy the Module 13 agent with cost/latency monitoring and a kill switch.
- ☁ **Cloud (CD stage 3):** deploy the agent to **Cloud Run** (serverless containers, scale-to-zero — you pay only on invocation). Actions deploys automatically on a passing build, authenticating via **Workload Identity Federation** (keyless — the modern practice; signals current rather than deprecated service-account-key habits). Define the Cloud Run service and Cloud SQL instance in **Terraform** (infra-as-code — a genuine differentiator). Observe via **Cloud Logging / Cloud Monitoring** — the same observability Module 14 wants, seen from the infra side.

### Module 16 — Capstone & Research Fluency
- ★◆ Capstone: an agentic system that takes a molecular design goal → generates candidates (Modules 5/10) → screens them with a pretrained MLIP tool (Module 9) → retrieves relevant literature (RAG, Module 12) → wrapped in a proper eval harness (Module 14) and deployed with monitoring (Module 15). Demonstrates the entire stack — the "shipped a real agent" evidence employers screen for.
- ★ Research fluency: reading/reproducing papers across both fields; tracking a fast-moving frontier; identifying load-bearing claims.
- ○ Reproduce one non-trivial result end to end at the chemistry/generative intersection.

---

## Cloud & CI/CD Track (GCP) ☁

Runs alongside the whole syllabus. Principle: acquire each infra skill exactly when an artifact justifies it — never front-load a pipeline onto something that doesn't need one (that's procrastination dressed as productivity). GCP chosen for existing familiarity; concepts transfer to AWS/Azure, and depth in one cloud reads better than shallow exposure to three. **Colab is a disposable GPU scratchpad only — it teaches none of the production skills that are the actual hiring signal.**

**The arc, end to end:**
1. **Foundation (from Module 2):** Git/GitHub + GitHub Actions running lint + pytest on push. Establishes the CI reflex and a clean commit history.
2. **Training (from Module 8):** Vertex AI Workbench for controlled-GPU notebooks, or Compute Engine spot/preemptible GPU instances for cheaper raw time and more real cloud mechanics. Billing alerts on day one.
3. **Data (Module 12):** Cloud Storage for corpora/artifacts; pgvector on Cloud SQL for the vector DB.
4. **CI stage 2 (Module 13):** Dockerize the agent + MCP server; Actions builds and pushes images to Artifact Registry.
5. **CD stage 3 (Module 15):** Auto-deploy to Cloud Run via Actions, authenticated with Workload Identity Federation (keyless); infra defined in Terraform; observability via Cloud Logging / Monitoring.
6. **Infra capstone (after Module 16):** redeploy the *identical container image* — no app code changes — to **GKE**. Because the artifact is unchanged, this is pure infra learning in isolation.

**The GKE redeployment (the infra capstone):**
Same image, deployed at increasing levels of control — Cloud Run (serverless) → **GKE Autopilot** (real Kubernetes objects, Google runs the control plane and node provisioning) → optionally GKE Standard (you manage node pools). **Most should stop at Autopilot** — the sweet spot of real-Kubernetes resume signal without the node-management tax; Standard matters mainly for infra-heavy platform roles.

What the redeploy teaches that Cloud Run hides (the interview-able Kubernetes concepts): pods and deployments (unit of work + desired-state controller), services and ingress (how traffic gets in — Cloud Run just hands you a URL), horizontal pod autoscaling (explicit, vs. Cloud Run's silent version), resource requests/limits (you declare CPU/memory — and own cost control), and config/secrets as first-class objects.

Forward-looking discipline that makes this an *addition* not a *migration*: keep the Cloud Run infra in Terraform (planned anyway), so adding a GKE target is a new infra module, not a teardown.

⚠️ **Cost discipline (part of the curriculum, not a footnote):** GPUs and always-on services bill fast, and a GKE cluster bills continuously — control plane and any minimum nodes cost money whether or not the agent is invoked. Use spot instances for training, scale-to-zero on Cloud Run, billing alerts from day one, and tear down anything idle. **Time-box the GKE phase:** stand it up, do the redeploy, capture a writeup/screenshot for the portfolio, tear the cluster down — don't leave it running as a trophy. Autopilot helps (no paying for idle node capacity). Learning to *not* burn money is itself the cost-optimization skill Module 15 screens for.
