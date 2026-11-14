# Standalone Skill ≠ Specialist Value: Stage-Gated RL + AlignTune DPO Across Llama-3.1-8B and Aya-Expanse-8B for Autonomous Red-Teaming

Autonomous red-teaming in realistic enterprise simulations like CybORG CAGE-4 shows a clear split between Reinforcement Learning agents and Large Language Models:
- **Pure RL (FiniteStateRedAgent):** Fast and cheap (3.34s–3.35s per 120-step episode, 0 LLM tokens). It is good at subnet discovery and lateral movement, but gets stuck at local privilege escalation (63.2%–65.3% *P(Root | Comp)*) and post-root disruption (61.5%–62.9% *P(Impact | Root)*). That leaves only **15.7%–15.8% end-to-end host disruption** (3.17–3.20 impacted hosts per episode).
- **Pure LLMs (Llama-3.1-8B-Instruct / Aya-Expanse-8B):** Each model fails in a **different way**, and both reach a **0.0% Impact Rate**:
  - **Llama-3.1-8B-Instruct** wanders across the network and burns **72,317 tokens/episode** (120.38s/episode). It compromises 24.9% of discovered hosts and roots 49.4% of those.
  - **Aya-Expanse-8B** hits a **92.7% conditional compromise rate**, but only over **82 discovered hosts in 30 episodes** (2.73 hosts/episode, against 20.0 for Pure RL). It keeps exploiting the first contractor-subnet IPs and almost never escalates (**1.3% root conversion, 1/76**).

This post introduces the **Adaptive Kill-Chain Router (AKCR)**, a stage-gated *hybrid RL + DPO* architecture built with the *Lexsi Labs open-source stack* (`aligntune`, `curatorkit`, `safetune`). We ran it on **30 controlled CAGE-4 seeds (120-step horizons, NVIDIA L4 GPU)** with **two different 8B model families**:

| Headline Metric | Llama-3.1-8B-Instruct (Hybrid DPO) | Aya-Expanse-8B (Hybrid DPO) |
| :--- | :---: | :---: |
| **Privilege Escalation** *P(Root \| Comp)* | **90.0%** vs 65.3% Pure RL (z = +6.22, p < 0.0001) | **91.2%** vs 63.2% Pure RL (z = +6.88, p < 0.0001) |
| **Post-Root Weaponization** *P(Impact \| Root)* | **98.9%** vs 61.5% (z = +9.03, p < 0.0001) | **98.9%** vs 62.9% (z = +8.71, p < 0.0001) |
| **End-to-End Yield** *P(Impact \| Disc)* | **32.1%** vs 15.7% (**2.05×**, 6.27 vs 3.20 hosts/ep) | **30.8%** vs 15.8% (**1.94×**, 6.13 vs 3.17 hosts/ep) |
| **Token Reduction vs Pure LLM** | **97.0%** (2,172 vs 72,317) | **84.8%** (1,988 vs 13,106) |
| **Speedup vs Pure LLM** | **10.1×** (11.89s vs 120.38s) | **3.66×** (12.13s vs 43.92s) |

**The main finding:** an LLM's standalone benchmark performance **does not predict** how useful it will be as a stage-specialized part of a hybrid agent. On their own, Llama and Aya behave in opposite ways and both fail. Inside AKCR they end up in nearly the same high-performing regime.

---

## 1. Architecture: Inverted Stage-Gated Routing
Most hybrid agent designs put an LLM at the top of the stack as a slow "planner" that calls RL or tools on every turn. AKCR **inverts** this hierarchy:
1. **Tactical Network Operator (RL):** `FiniteStateRedAgent` drives **91.9%–92.8% of all steps** (`110.3–111.4 / 120`). It handles subnet reconnaissance (`DiscoverRemoteSystems`), initial exploitation (`ExploitRemoteService`), and post-root service degradation (`DegradeServices`).
2. **Stage-Gated Trigger:** `AdaptiveKillChainRouter` tracks parsed CybORG host states (`Discovered`, `Compromised`, `Rooted`). It hands control to the LLM specialist only when a host is stuck at an unprivileged foothold ($\text{UID} \neq 0$, attempts `< 5`).
3. **Constrained Candidate Menu:** The LLM never sees the full network action space. It gets a **3-action menu** for the target host: `PrivilegeEscalate <host>`, `DegradeServices <host>`, `Sleep`.
4. **Strict Session Anchoring & Handoff:** The router binds the selected action to CybORG's source `RedAbstractSession` (`session=0`) and gives control straight back to RL.

Escalations usually succeed on the first attempt, so each one takes about 1 LLM step and about 2 handoffs (`RL → LLM → RL`). Average LLM usage stays at **8.6 (Aya) to 9.7 (Llama) steps per 120-step episode, or 7.2%–8.1% of the interaction budget.**

---

## 2. CybORG-Native DPO Curation & Alignment with the Lexsi Stack
Off-the-shelf instruction-tuned models often fail in cyber simulators for two reasons: **conversational chatter** (*"Sure! Let's inspect `/etc/passwd` first..."*) and **hallucinated raw bash** (`sudo -l`, `chmod +s /bin/bash`). Neither is a valid action in CybORG's action space.

To ground the specialist without brittle prompt hacks, we built a CybORG-native preference pipeline with the Lexsi stack:
- **[`curatorkit`](https://github.com/Lexsi-Labs/CuratorKIT) & [`safetune`](https://github.com/Lexsi-Labs/SafeTune) (`src/training/prepare_dpo_data.py`):** **267 preference pairs** from two sources:
  - `180` multi-subnet enterprise topology pairs covering all 8 CAGE-4 subnets.
  - `87` live CAGE-4 self-play rollout pairs: `29` distractor-action rejections, `29` conversational-chatter rejections, and `29` raw-bash-hallucination rejections.
  - The `curatorkit==0.1.22` + `safetune==0.1.12` gate checks **100.0% schema completeness**, **100.0% `PrivilegeEscalate <host>` syntax compliance**, and **0 fabricated kernel tokens or raw bash payloads**. It writes a SHA-256 provenance manifest (`dpo_curation_audit.json`, dataset sha256 `1b8a96b3…c131ca`).
- **[`aligntune`](https://github.com/Lexsi-Labs/aligntune) (`src/training/train_dpo.py`):** 4-bit NormalFloat (`nf4`) LoRA DPO adapters (`r=16, alpha=32, beta=0.1, lr=5e-6`, 3 epochs) trained via `aligntune==0.1.35`. Each model's own chat template is used during training (Cohere `<|START_OF_TURN_TOKEN|>` format for Aya with its **8K context window**, Llama-3.1 header format for Llama), so training prompts match inference prompts exactly.

---

## 3. Empirical Results: 4-Quadrant Ablation (30 Seeds, 120 Steps, L4 GPU)
To separate **structural routing gains** from **post-training alignment gains**, we evaluated four configurations on the same 30 random seeds (`101–130`):
1. **Baseline Pure RL** (`FiniteStateRedAgent`)
2. **Baseline Pure LLM** (the LLM drives every step of the 120-step episode)
3. **Hybrid + Frozen Base** (AKCR routing + base instruct model, no fine-tuning)
4. **Hybrid + AlignTune DPO** (AKCR routing + `aligntune` LoRA DPO adapter)

### 3.1 `meta-llama/Llama-3.1-8B-Instruct`
| Configuration | Ep | P(Comp \| Disc) | P(Root \| Comp) | P(Impact \| Root) | End-to-End Yield | Avg Impacted Hosts/Ep | RL / LLM Steps | Avg Tokens | Avg Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline Pure RL** | 30 | 39.0±5.3% | 65.3±22.9% | 61.5±23.5% | 15.7% (`96/613`) | 3.20 | 120.0 / 0.0 | 0 | 3.35s |
| **2. Baseline Pure LLM** | 30 | 24.9±9.9% (`87/349`) | 49.4±26.9% (`43/87`) | 0.0±0.0% | 0.0% (`0/349`) | 0.00 | 0.0 / 120.0 | 72,317 | 120.38s |
| **3. Hybrid + Frozen Base** | 30 | 36.8±5.5% | 89.2±14.7% | 98.4±4.5% | 32.3% (`186/576`) | 6.20 | 110.3 / 9.7 | 2,180 | 11.89s |
| **4. Hybrid + AlignTune DPO** | 30 | 36.1±5.7% | **90.0±14.0%** | **98.9±3.3%** | **32.1%** (`188/585`) | **6.27** | **110.3 / 9.7** | **2,172** | **11.89s** |

![](https://github.com/Viraj465/Red-Team-Agents-LexsiLabs/blob/main/results/meta_llama_llama_3_1_8b_instruct_v2/killchain_meta_llama_llama_3_1_8b_instruct_chart.png)

### 3.2 `CohereLabs/aya-expanse-8b`
| Configuration | Ep | P(Comp \| Disc) | P(Root \| Comp) | P(Impact \| Root) | End-to-End Yield | Avg Impacted Hosts/Ep | RL / LLM Steps | Avg Tokens | Avg Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline Pure RL** | 30 | 39.8±5.0% | 63.2±19.1% | 62.9±21.0% | 15.8% (`95/600`) | 3.17 | 120.0 / 0.0 | 0 | 3.34s |
| **2. Baseline Pure LLM** | 30 | 92.7±8.8% (`76/82`) | 1.3±3.6% (`1/76`) | 0.0±0.0% | 0.0% (`0/82`) | 0.00 | 0.0 / 120.0 | 13,106 | 43.92s |
| **3. Hybrid + Frozen Base** | 30 | 34.1±7.2% | **91.2±11.3%** | **98.9±4.3%** | **30.8%** (`184/598`) | **6.13** | 111.4 / 8.6 | 1,988 | 12.01s |
| **4. Hybrid + AlignTune DPO** | 30 | 34.1±7.2% | **91.2±11.3%** | **98.9±4.3%** | **30.8%** (`184/598`) | **6.13** | **111.4 / 8.6** | **1,988** | 12.13s |

![](https://github.com/Viraj465/Red-Team-Agents-LexsiLabs/blob/main/results/coherelabs_aya_expanse_8b_v2/killchain_coherelabs_aya_expanse_8b_chart.png)

### 3.3 Two-Proportion Z-Tests & Episode-Clustered 95% Bootstrap CIs
| Comparison | Metric | Llama-3.1-8B-Instruct | Aya-Expanse-8B |
| :--- | :--- | :---: | :---: |
| **Hybrid DPO vs Pure RL** | P(Root \| Comp) | `190/211` (**90.0%**, CI `[85.1%, 95.1%]`) vs `156/239` (65.3%) → **z = +6.22, p < 0.0001** | `186/204` (**91.2%**, CI `[87.1%, 95.2%]`) vs `151/239` (63.2%) → **z = +6.88, p < 0.0001** |
| **Hybrid DPO vs Pure RL** | P(Impact \| Root) | `188/190` (**98.9%**, CI `[97.4%, 100.0%]`) vs `96/156` (61.5%) → **z = +9.03, p < 0.0001** | `184/186` (**98.9%**, CI `[97.3%, 100.0%]`) vs `95/151` (62.9%) → **z = +8.71, p < 0.0001** |
| **Hybrid DPO vs Pure RL** | End-to-End Yield | `188/585` (**32.1%**) vs `96/613` (15.7%) → **z = +6.70, p < 0.0001 (2.05×)** | `184/598` (**30.8%**) vs `95/600` (15.8%) → **z = +6.12, p < 0.0001 (1.94×)** |
| **Hybrid DPO vs Hybrid Frozen** | P(Root \| Comp) | `190/211` vs `189/212` → z = +0.30, p = 0.7627 | `186/204` vs `186/204` → z = 0.00, p = 1.0000 |

> **Note on statistics:** The z-tests pool host-level transitions across episodes. The episode-clustered bootstrap CIs resample whole episodes (1,000 resamples, N = 30), which accounts for correlation between hosts within the same episode.

---

## 4. What the Two Models Teach Us

### 4.1 Standalone capability does not predict specialist value
| Pure LLM Profile | Llama-3.1-8B-Instruct | Aya-Expanse-8B |
| :--- | :---: | :---: |
| Hosts discovered (30 eps) | 349 | **82** |
| P(Comp \| Disc) | 24.9% | **92.7%** |
| P(Root \| Comp) | **49.4%** | 1.3% |
| Impact Rate | 0.0% | 0.0% |
| Tokens / episode | 72,317 | **13,106** |

On their own, the two models fail in opposite ways. Llama explores widely and expensively. Aya stays local and cheap, and almost never escalates. **Inside AKCR, both reach 90–91% root conversion and ~31–32% end-to-end yield.** The router takes away the part each model is worst at (long-horizon network planning) and gives it the one decision 8B instruct models handle reliably: picking the escalation action for a specific stuck host.

### 4.2 Pure Aya's 92.7% is a small-denominator effect
Pure Aya's 92.7% *P(Comp | Disc)* does **not** mean it is better at exploration. It comes from 76 compromises out of only 82 discovered hosts. Pure Aya keeps choosing `ExploitRemoteService` on the first subnet's IPs and rarely runs `DiscoverRemoteSystems`, so its serialized action menu stays short. That is also why it uses **5.5× fewer tokens than Pure Llama**, which discovers more hosts and gets a much longer candidate list every step.

### 4.3 Episode Impact Rate saturates, so we report End-to-End Yield
Pure RL and every hybrid configuration reach a **100% episode Impact Rate**, because one disrupted host is enough to count. That metric hides the gap between **3.17–3.20** and **6.13–6.27** impacted hosts per episode. **End-to-End Yield P(Impact | Disc)** shows the real ~2× difference in mission lethality.

### 4.4 Efficiency is a Pareto trade-off, not a free lunch
- **Compared with Pure LLM:** 84.8%–97.0% fewer tokens and 3.66×–10.1× faster, because the LLM is used on only 7.2%–8.1% of steps.
- **Compared with Pure RL:** Hybrid execution is about 3.6× slower (12s vs 3.3s per episode). In return, it gains +24.7 to +28.0 pp in root conversion and roughly doubles the number of impacted hosts.

### 4.5 Frozen vs DPO: no observed difference under constrained routing
On Aya, Hybrid Frozen and Hybrid DPO produced **identical** task metrics (same host counts, same 1,988 tokens, same 133.7 completion tokens per episode). On Llama the gap was +0.8 pp (p = 0.76). This is a ceiling effect, not proof that DPO is useless:
- The router reduces each decision to a **3-choice menu** with **greedy decoding** (`do_sample=False`).
- Preference-tuned 8B models already output the exact `PrivilegeEscalate <hostname>` string zero-shot (~15.5 completion tokens per call, no chatter).
- Same seeds plus the same greedy output give the same simulator trajectory.

We describe this as **no observed performance difference**, not as statistical equivalence. To verify that the adapter really changes model outputs, the pipeline now:
- logs the **adapter SHA-256**, `model_mode`, and a **per-reply hash** to `logs/raw_llm_audit.jsonl`;
- keeps the LoRA adapter **unmerged** on the 4-bit base, to avoid rounding errors from 4-bit merge;
- seeds Python/NumPy RNGs per episode;
- records a code fingerprint when `git_sha` is unavailable on Colab.

---

## 5. Honest Takeaways for Agent Builders
1. **Environment grounding matters more than prompting alone:** Early on, passing the target host's session ID instead of CybORG's source `RedAbstractSession` (`session=0`) silently broke every LLM escalation. Fixing session resolution and syncing RL's internal state on router-intercepted turns unlocked the jump to ~90% *P(Root | Comp)*.
2. **Scaffolding sets the ceiling:** Once the stage-gated router limits the LLM to candidate CybORG actions, strong 8B instruct models already reach 89–91% root conversion zero-shot. The architecture, not the fine-tuning, produces the main capability gain.
3. **DPO is reliability insurance:** In a well-scaffolded agent, AlignTune DPO's main value is enforcing deterministic syntax, no bash hallucinations, and single-turn execution. Those protections matter more as prompts get less constrained.
4. **Don't pick a model by its standalone score:** Aya was the worst standalone escalator (1.3%) and the best hybrid escalator (91.2%).

### What's next
- **Fixed-schedule vs adaptive routing control**, to isolate how much the router's state-triggered handoff contributes.
- **LLM allocation sweep** (0% / 5% / 10% / 15% / 20% of steps) under locked seeds.
- **DPO on less constrained menus** (full candidate lists, sampling decoding), where the frozen model has room to drift.
- **Held-out topology evaluation** to test generalization beyond the training distribution.

---

## 6. Try It Yourself
All code, unit tests (`14/14` passing), curation audit manifests, provenance files, and the one-click Google Colab L4 GPU notebook (`LexsiLabs_Colab_GPU.ipynb`) are open-source:
- **GitHub Repository:** [https://github.com/Viraj465/Red-Team-Agents-LexsiLabs](https://github.com/Viraj465/Red-Team-Agents-LexsiLabs.git)
- **Lexsi Labs Ecosystem:** [`aligntune`](https://github.com/Lexsi-Labs/aligntune), [`curatorkit`](https://github.com/Lexsi-Labs/CuratorKIT), [`safetune`](https://github.com/Lexsi-Labs/SafeTune)
- **Models:** [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [`CohereLabs/aya-expanse-8b`](https://huggingface.co/CohereLabs/aya-expanse-8b)
- **Benchmark:** CybORG CAGE Challenge 4

```bash
python scripts/run_colab_pipeline.py --model-name CohereLabs/aya-expanse-8b --episodes 30 --max-steps 120
python scripts/run_colab_pipeline.py --model-name meta-llama/Llama-3.1-8B-Instruct --episodes 30 --max-steps 120
```
