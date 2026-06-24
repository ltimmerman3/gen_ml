# Learning Plan — Generative AI, Computational Chemistry & Agentic Engineering

**Pacing:** ~10 hrs/week · strict theory/build alternation (one week theory, next week build).
**Honest timeline:** full ★-spine ≈ 9–11 months; **employable agentic competence ≈ month 5–6** (the job goal arrives before the syllabus finishes).
**Keyed to:** placement quiz results (strengths: equivariance/group theory, physics-grounded reasoning, reparameterization, agentic *judgment*; gaps: DL canon, transformers/LLMs, diffusion, agentic *vocabulary*, measured RAG).

**Legend:** ★ job-critical · ◆ chemistry · ○ enrichment · ☁ cloud/CI-CD. "T" = theory week, "B" = build week.

---

## Strategy (why this order)

Three principles drawn from your quiz:

1. **Frontload the highest-leverage gaps, not the syllabus order.** Your weakest *foundational* spot is classical-ML literacy (bias-variance) and the DL canon (backprop dimensions, norms, autograd internals). These gate everything downstream, so Phase 1 closes them first.
2. **Convert judgment into vocabulary fast.** Your agentic *judgment* (quiz 5.4) is already strong; only the vocabulary and wiring are missing. So the job-track (RAG/agents/eval) starts earlier than its module number suggests — it's relabeling, not concept-building.
3. **Spend lightly where you're already strong.** Equivariance/group theory (1.3, 1.4) and the physics framing (2.4) get compressed to "tighten and confirm," buying time for the gaps.

---

## Phase 0 — Setup (Week 0, ~5 hrs, one-time)

- ☁ Git/GitHub repo for the whole journey; one monorepo with a folder per module.
- ☁ GitHub Actions: lint + pytest on push (trivial now, reflex for later).
- Local Python env (uv or conda), PyTorch installed, GPU access plan confirmed (laptop fine for Phase 1).
- Drop the syllabus + quiz + this plan into the repo's `/docs`.

---

## PHASE 1 — Foundations & Closing the Core Gaps (Weeks 1–8, months 1–2)

*The highest-leverage block. Targets the exact gaps the quiz exposed: bias-variance, KL directionality, backprop dimensions, norms, autograd internals.*

**Week 1 (T) — Math gaps: information theory.** ★ Module 1 info-theory strand, expanded. KL divergence directionality (the 1.1 gap): forward = mode-covering/mean-seeking, reverse = mode-seeking, *why VAE samples are blurry*. Entropy, cross-entropy, ELBO as KL decomposition. Confirm/tighten the equivariance strand (you're already strong — just the "transform within fixed ℓ / no mixing across ℓ" point from 1.4). *Deliverable:* short notes deriving both KL directions and predicting which models use which.

**Week 2 (B) — Autograd from scratch.** ★ Module 2 build. NumPy reverse-mode autodiff engine + MLP. This single build fixes quiz gaps 2.2 (backprop dimensions — get the transposes right: ∂L/∂W = δy·xᵀ, ∂L/∂x = Wᵀδy), 6.2, and 6.4 mechanics at once. *Deliverable:* working micrograd-style engine; gradients match PyTorch numerically.

**Week 3 (T) — Classical-ML literacy + DL core theory.** ★ Module 2 theory. The flagged gap: bias-variance tradeoff properly, then regularization, double descent (why overparameterization isn't a contradiction). Then init (Xavier/He), and the norms (the 2.3 gap): batch vs. layer norm normalize *activations* over different axes; layer norm is per-example over features, which is why transformers use it. Residual connections. *Deliverable:* one-page written explanation of bias-variance → double descent in your own words.

**Week 4 (B) — PyTorch port + DL mechanics.** ★ Port the Week-2 engine to PyTorch; build reusable layers. Nail the 6.2 distinction in code: `model.eval()` (layer behavior) vs. `torch.no_grad()` (graph-building). Train a small MLP on a real dataset end to end. *Deliverable:* clean PyTorch training loop you understand line by line.

**Week 5 (T) — Probability/estimation + geometric DL (light).** ★◆ Module 1 estimation strand (MLE↔KL, score matching — seeds diffusion later) + change-of-variables (you derived 1.2 correctly; formalize it — it's the flows backbone). ◆ Module 3 conceptually: message passing, invariance vs. equivariance (you're strong here — confirm and move on). *Deliverable:* notes connecting change-of-variables → normalizing-flow likelihood.

**Week 6 (B) — First generative build: VAE.** ★◆ Module 5 build, anchored on your existing strength. You already own the reparameterization trick (quiz 3.2) — lead with it. Build a VAE; induce and then fix posterior collapse (the 3.1 gap, made concrete). *Deliverable:* working VAE on a simple dataset; latent-space visualization; a deliberately-collapsed-then-fixed run.

**Week 7 (T) — VAE theory + chemistry on-ramp.** ★◆ Module 5 theory fully (ELBO terms, β-VAE, VQ-VAE discrete latents). ◆ Molecular representations: SMILES/SELFIES, when discrete latents matter for molecules. *Deliverable:* notes; pick the molecular dataset you'll use later (e.g., QM9-scale).

**Week 8 (B) — Molecular VAE.** ◆ Module 5 chemistry practice: train a VAE to encode/generate valid molecules via SMILES/SELFIES. First real chemistry deliverable — resume-legible. *Deliverable:* molecular VAE; report validity/uniqueness of generated structures.

> **Phase 1 milestone:** core DL gaps closed; two working generative builds; autograd internalized. You now have the foundation the transformer build needs.

---

## PHASE 2 — The Transformer & the LLM Bridge (Weeks 9–16, months 3–4)

*The hinge of the whole plan. Module 4 → 11 is where the job track is built. You start near-zero here (quiz Strand 4) but with good instincts — expect it to click faster than the week count suggests.*

**Week 9 (T) — Attention & transformer theory.** ★ Module 4 theory. Self-attention from scratch: Q/K/V, the √dₖ scaling (the 4.1 gap: large dot products saturate softmax → vanishing gradients). Multi-head, the full block. Positional encodings: sinusoidal vs. RoPE (the 4.2 gap). *Deliverable:* annotated diagram of one transformer block you can explain end to end.

**Week 10 (B) — GPT from scratch.** ★ Module 4 build — *non-negotiable for the job*. Implement and train a decoder-only GPT on a small corpus. Tokenization (BPE), sampling (temperature/top-k/nucleus). *Deliverable:* a working small GPT you trained yourself; generate samples.

**Week 11 (T) — LLMs in depth.** ★ Module 11 theory. Pretraining → post-training → RLHF/DPO (concept). Scaling laws. Context windows and why long context ≠ memory (the 4.4 gap — your "recall on demand" instinct, made precise: effective-context degradation + cost/latency). KV-cache properly (the 4.3 gap: cache past K/V to avoid recompute; memory scales linearly with seq-len × layers × heads). *Deliverable:* notes; a crisp written answer to "why RAG despite 1M context."

**Week 12 (B) — Fine-tuning hands-on.** ★ Module 11 build. LoRA/QLoRA on an open model for a domain task. The fine-tune-vs-prompt-vs-retrieve decision in practice. ☁ first real GPU compute (Vertex AI Workbench or a Compute Engine spot GPU; billing alerts on). *Deliverable:* a fine-tuned model + a written rationale for why fine-tuning (vs. RAG) was right here.

**Week 13 (T) — RAG & context engineering theory.** ★ Module 12 theory — directly upgrades your quiz-5.2 gap. Embeddings, chunking *strategies* (not arbitrary — the thing you flagged), hybrid search, re-ranking, and **retrieval-quality metrics** (the measurement you skipped: recall@k, MRR, faithfulness). *Deliverable:* notes specifying how you'll *measure* retrieval this time.

**Week 14 (B) — Measured RAG over a chemistry corpus.** ★◆ Module 12 build. Build RAG over chemistry papers/datasets and **measure it honestly** — the explicit fix for quiz 5.2. ☁ Cloud Storage + pgvector on Cloud SQL. *Deliverable:* RAG system with a retrieval-metrics report (this is the artifact your prototype lacked).

**Week 15 (T) — Agents & MCP theory.** ★ Module 13 theory — converts your strong agentic *judgment* into *vocabulary*. ReAct loop (the 5.1 gap), tool/function-calling design, MCP architecture (you nailed the *why* in 5.3 — now the *how*: tools/resources/prompts), single vs. multi-agent honestly. *Deliverable:* design doc for the agent you'll build next week.

**Week 16 (B) — Build an agent with MCP.** ★◆ Module 13 build. An agent that calls real tools — a pretrained MLIP, RDKit, a literature DB — via an MCP server *you* build. ☁ Dockerize + push to Artifact Registry. *Deliverable:* working tool-using agent; your first MCP server.

> **Phase 2 milestone (≈ month 4):** you can build, fine-tune, and deploy LLM systems and tool-using agents. This is the point where you're becoming *interview-able* for agentic roles. Start light job-market reconnaissance here.

---

## PHASE 3 — Production, Eval & Employability (Weeks 17–22, months 5–6)

*Turns "builds agents" into "ships and evaluates agents" — the actual hiring signal. Your quiz-5.4 answer shows the judgment is already here; this phase adds the engineering scaffolding.*

**Week 17 (T) — Evaluation & observability theory.** ★ Module 14 — the single biggest hiring signal, and where your 5.4 judgment becomes formal method. Eval design for prompts/RAG/agents; LLM-as-judge and its pitfalls (you already flagged hallucination + "don't know the right observable"); tracing, online vs. offline. *Deliverable:* eval design doc for your Week-16 agent — expand your quiz-5.4 answer into a real harness spec.

**Week 18 (B) — Build the eval harness.** ★ Module 14 build — *portfolio gold*. Offline test suite + observability for the agent. Catch a real failure mode (wrong tool selection / loop / cost blowup). *Deliverable:* eval harness with a written failure-mode analysis.

**Week 19 (T) — Production theory.** ★ Module 15. Cost optimization (caching, routing, model selection — the 40–70% lever), latency/streaming, failure modes under load, deployment patterns. *Deliverable:* a cost/latency budget for your agent.

**Week 20 (B) — Deploy for real.** ★☁ Module 15 build. Deploy the agent to Cloud Run via GitHub Actions; Workload Identity Federation (keyless); Terraform for the infra; Cloud Logging/Monitoring. *Deliverable:* a live, monitored, auto-deployed agent with a kill switch.

**Week 21 (T) — Capstone design + research fluency.** ★◆ Module 16 planning. Design the intersection capstone: molecular design goal → generate candidates → screen with pretrained MLIP → retrieve literature → eval harness → deployed. Begin a paper-reading habit. *Deliverable:* capstone spec + architecture diagram.

**Week 22 (B) — Capstone build (part 1) + portfolio.** ★◆ Start the capstone; stand up the skeleton end to end. Begin writing it up for a portfolio/README. *Deliverable:* working skeleton of the full-stack agentic-chemistry system.

> **Phase 3 milestone (≈ month 6): employable.** You have a deployed, evaluated, monitored agentic system with a chemistry differentiator and a CI/CD story. This is the "shipped a real agent" evidence. **Begin applying.**

---

## PHASE 4 — Depth, Diffusion & the Full Spine (Weeks 23+, months 6–11)

*Run in parallel with job-searching. Completes the theoretical spine and the chemistry frontier. These deepen you from "can ship" to "can innovate" — your stated end goal — but aren't gating for the offer.*

- **Diffusion block (Module 8):** ★ DDPM from scratch (greenfield per quiz 3.3/3.4), then score-SDEs and the probability-flow ODE. T/B alternation, ~4 weeks. The biggest remaining theory gap.
- **Flows & EBMs/GANs (Modules 6–7):** ○ Normalizing flows (your 1.2 derivation pays off), EBMs, GANs. ~3 weeks, enrichment.
- **Generative chemistry (Module 10):** ◆ Equivariant diffusion for 3D conformers; property-conditioned generation; retrosynthesis transformer. ~3 weeks. This is where your equivariance strength + diffusion meet.
- **MLIPs deeper (Module 9):** ◆ Optional from-scratch SchNet on rMD17 (your 2.4 answer shows you're ready); otherwise stay at "use pretrained MACE-MP." ~2 weeks.
- **Capstone completion + GKE infra capstone:** ★☁ Finish the capstone; redeploy the identical image to GKE Autopilot as the infra capstone (time-boxed — tear it down after). ~3 weeks.
- **Research fluency (Module 16):** ○ Reproduce one non-trivial result at the chemistry/generative intersection.

---

## Cadence & guardrails

- **Weekly rhythm:** ~10 hrs = roughly 4 hrs focused study + 5 hrs build/practice + 1 hr notes/commit. Theory weeks lean reading/derivation; build weeks lean code.
- **Every build week ends with a commit + a short README.** The repo *is* the portfolio; treat it as such from Week 0.
- **Re-quiz checkpoints:** after Phase 1 (week 8) and Phase 2 (week 16), spot-check the gaps this plan targeted (KL directionality, backprop dims, retrieval metrics, ReAct/MCP) to confirm they closed.
- **Cost discipline is curriculum:** spot GPUs, scale-to-zero, billing alerts, tear down idle resources. (Also the Module 15 skill employers screen for.)
- **The job goal does not require Phase 4.** If an offer lands at month 6, Phase 4 becomes professional development, not a prerequisite. Don't let "finish the syllabus" delay applying.

---

## Quiz-gap → plan-week traceability

| Quiz gap | Closed in |
|---|---|
| 1.1 KL directionality | Week 1 |
| 1.4 "within-ℓ" irrep point | Week 1 (tighten) |
| 2.1 bias-variance / double descent | Week 3 |
| 2.2 backprop dimensions/transposes | Week 2 |
| 2.3 batch vs. layer norm | Week 3 |
| 3.1 ELBO / posterior collapse | Weeks 6–7 |
| 3.3–3.4 DDPM / score-SDEs | Phase 4 (diffusion block) |
| 4.1 √dₖ saturation | Week 9 |
| 4.2 RoPE | Week 9 |
| 4.3 KV-cache scaling | Week 11 |
| 4.4 long-context ≠ memory | Week 11 |
| 5.1 ReAct loop | Week 15 |
| 5.2 measured retrieval (the big one) | Weeks 13–14 |
| 6.2 eval() vs. no_grad() | Week 4 |
| 6.4 double-backward / MLIP training | Week 2 + Phase 4 |
