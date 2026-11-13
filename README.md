# 97% Fewer Tokens, 2x Higher Kill Chain Yield: Stage Gated RL + AlignTune DPO for Autonomous Red Teaming Agents

Autonomous Red-teaming in realistic enterprise simulations like CybORG CAGE-4 exposes a sharp capability shift between the Reinforcement Learning models and Large Language Models:
- Pure RL (FiniteStateRedAgent): This are fast and cheap (3.35s/120-step episode, 0 LLm tokens) and excels at subnet discovery and lateral movement, but stalls at localized privilege escalation (65.3% *P(Root | Comp)*) and post root disruption (61.5% *P(Impact | Root)*), yielding only 15.7% end-to-end host disruption.
- Pure LLMs (llama-3.1-8b-it / Qwen2.5-7b-it) They particularly understands escalation semantics but results in heavy burning of tokens (**40,832 to 72,317** tokens/episode) strolling across 120 step horizon (120s-140s /episode) and achieving **0.0% Impact Rate** which never completes the mission.

This blog introduces **Adaptive Kill-Chain Router (AKCR)** which is a stage-gated *hybrid RL + DPO* architecture built using *Lexsi Labs Opensource stack (aligntune, curatorkit, and safetune). Across **30 distinct CAGE-4 random seeds (120-step horizons)**, AKCR with **llama-3.1-8b-it** achieving:
- **90.0% Privilege Escalation** *(P(Root | Comp)* against 65.3% Pure RL, z = +6.22, p < 0.0001).
- **98.9% Post-Root weaponization** *(P(Impact | Root)* against 61.5% Pure RL, z = +9.03, p < 0.0001).
- **2.05x end-to-end host disruption yield** (32.1% / 6.27 impacted host/ episode vs 15.7% / 3.20 hosts in Pure RL, z = +6.70, p < 0.0001).
- **97.0% token reduction & 10.1x speedup** over Pure LLM (2,172 vs 72,317 tokens/ episode; 11.89s vs 120.38s).
---

## 1. Architecture: Inverted Stage-Gated Routing
Most hybrid agent designs put an LLM at the top of the stack as a slow "planner" that calls RL or tools on every turn. AKCR **inverts** this hierarchy:
1. **Tactical Network Operator (RL):** `FiniteStateRedAgent` drives **91.9% of all steps** (`110.3 / 120`), handling continuous subnet reconnaissance (`DiscoverRemoteSystems`), initial exploitation (`ExploitRemoteService`), and post-root service degradation (`DegradeServices`).
2. **Stage-Gated Trigger:** `AdaptiveKillChainRouter` monitors parsed CybORG host states (`Discovered`, `Compromised`, `Rooted`). Only when a host is trapped at an unprivileged foothold ($\text{UID} \neq 0$, attempts `< 5`) does it hand control to the LLM specialist.
3. **Strict Session Anchoring & Handoff:** The LLM selects an exact CybORG action (`PrivilegeEscalate <hostname>`), which the router binds to CybORG's active source `RedAbstractSession` (`session=0`) before immediately returning control to RL.
Because 1-step DPO escalations typically root the target host on the first attempt and hand control right back to RL, each escalation takes ~1 LLM step and produces ~2 handoff transitions (`RL → LLM → RL`), keeping average LLM usage to just **9.7 steps (8.1%) per 120-step episode**.
---

## 2. CybORG-Native DPO Curation & Alignment with the Lexsi Stack
Off-the-shelf instruction-tuned models frequently fail in cyber simulators for two reasons: **conversational chatter** (*"Sure! Let's inspect `/etc/passwd` first..."*) and **raw bash hallucinations** (`sudo -l`, `chmod +s /bin/bash`), neither of which is a valid action in CybORG's action space.
To ground the specialist without brittle prompt hacks, we built a CybORG-native preference pipeline using the Lexsi Stack:
- **`curatorkit` & `safetune` (`src/training/prepare_dpo_data.py`):** We generated **267 preference pairs** (`197` unique prompts) from two sources:
  - `180` multi-subnet enterprise topology pairs spanning all 8 CAGE-4 subnets (`contractor_network`, `public_access_zone`, `office_network`, `admin_network`, `restricted_zone_a/b`, `operational_zone_a/b`).
  - `87` live CAGE-4 self-play rollout pairs (`29` distractor action rejections like `DegradeServices` on unrooted hosts, `29` conversational chatter rejections, and `29` raw bash hallucination rejections).
  - Our automated `curatorkit==0.1.22` + `safetune==0.1.12` gate verifies **100.0% schema completeness**, **100.0% `PrivilegeEscalate <host>` syntax compliance** on chosen actions, and **0 fabricated kernel tokens or raw bash payloads**, emitting a SHA-256 provenance manifest (`dpo_curation_audit.json`).
- **`aligntune` (`src/training/train_dpo.py`):** We post-trained 4-bit NormalFloat (`nf4`) LoRA adapters (`r=16, alpha=32, beta=0.1`) via `aligntune==0.1.31` so the specialist outputs deterministic, simulator-executable actions with zero preamble.
---

## 3. Empirical Results: 4-Quadrant Ablation (30 Seeds, 120 Steps, L4 GPU)
To isolate **structural routing gains** from **post-training alignment gains**, we evaluated four configurations across 30 identical random seeds:
1. **Baseline Pure RL** (`FiniteStateRedAgent`)
2. **Baseline Pure LLM** (LLM driving every step of the 120-step episode)
3. **Hybrid + Frozen Base** (AKCR routing + unaligned base instruct model)
4. **Hybrid + AlignTune DPO** (AKCR routing + `aligntune` LoRA DPO adapter)
### Primary Results (`meta-llama/Llama-3.1-8B-Instruct`)
| Configuration | Ep | P(Comp | Disc) | P(Root | Comp) | P(Impact |Root) | End-to-End Yield P(Impact | Disc) | Avg Impacted Hosts/Ep | RL / LLM Steps | Avg Tokens | Avg Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline Pure RL** | 30 | 39.0±5.3% | 65.3±22.9% | 61.5±23.5% | 15.7% (`96/613`) | 3.20 | 120.0 / 0.0 | 0 | 3.35s |
| **2. Baseline Pure LLM** | 30 | 24.9±9.9% | 49.4±26.9% | 0.0±0.0% | 0.0% (`0/349`) | 0.00 | 0.0 / 120.0 | 72,317 | 120.38s |
| **3. Hybrid + Frozen Base** | 30 | 36.8±5.5% | 89.2±14.7% | 98.4±4.5% | 32.3% (`186/576`) | 6.20 | 110.3 / 9.7 | 2,180 | 11.89s |
| **4. Hybrid + AlignTune DPO** | 30 | 36.1±5.7% | **90.0±14.0%** | **98.9±3.3%** | **32.1%** (`188/585`) | **6.27** | **110.3 / 9.7** | **2,172** | **11.89s** |

![](https://github.com/Viraj465/Red-Team-Agents-LexsiLabs/blob/main/results/meta_llama_llama_3_1_8b_instruct_v2/killchain_meta_llama_llama_3_1_8b_instruct_chart.png)

### Two-Proportion Z-Test Significance & Episode-Clustered 95% Bootstrap CIs
- **Hybrid DPO vs. Pure RL — P(Root | Comp):** `190/211` (**90.0%**, 95% CI `[85.1%, 95.1%]`) vs. `156/239` (`65.3%`) → **z = +6.22, p < 0.0001**
- **Hybrid DPO vs. Pure RL — P(Impact | Root):** `188/190` (**98.9%**, 95% CI `[97.4%, 100.0%]`) vs. `96/156` (`61.5%`) → **z = +9.03, p < 0.0001**
- **Hybrid DPO vs. Pure RL — End-to-End Yield P(Impact | Disc)$:** `188/585` (**32.1%**) vs. `96/613` (`15.7%`) → **z = +6.70, p < 0.0001** (**2.05× yield**)
- **Hybrid DPO vs. Hybrid Frozen — P(Root | Comp):** `190/211` (`90.0%`) vs. `189/212` (`89.2%`) → $z = +0.30, p = 0.7627$ (statistical parity, with DPO eliminating residual syntax/chatter drift)
---

## 4. Cross-Model Generalization & Key Engineering Lessons
We ran the same 30-seed, 120-step L4 GPU ablation on **`Qwen/Qwen2.5-7B-Instruct`** to test cross-architecture behavior:
- **Token Economics & Mission Completion Hold:** Hybrid routing achieved a **100.0% episode Impact Rate** (vs. `90.0%` Pure RL and `0.0%` Pure LLM) and cut tokens by **89.7%** (`4,207` vs. `40,832` tokens).
- **Post-Root Weaponization Improves Significantly:** Hybrid routing pushed $P(\text{Impact} \mid \text{Root})$ from `60.7%` (Pure RL) to **78.3%** (Frozen, $z = +3.03, p = 0.0025$) and **73.7%** (DPO, $z = +2.08, p = 0.0379$).
- **Two Honest Takeaways for Agent Builders:**
  1. **Environment Grounding > Prompting Alone:** Earlier in development, passing the target host's session ID instead of CybORG's source `RedAbstractSession` (`session=0`) silently broke LLM escalations. Fixing session resolution and synchronizing RL's internal state on router-intercepted turns unlocked the jump to `90.0%` P(Root | Comp).
  2. **Scaffolding Sets the Ceiling, DPO Locks In Reliability:** Once the stage-gated router constrains the prompt to candidate CybORG actions, strong 8B instruct models already reach `89.2%` (`Llama-3.1-8B`). AlignTune DPO's primary value in a well-scaffolded agent is enforcing deterministic syntax, zero bash hallucination, and single-turn execution—preventing the multi-step drift that destroys Pure LLM baselines.
---

## 5. Try It Yourself
All code, unit tests (`14/14` passing), curation audit manifests, and the one-click Google Colab L4 GPU notebook (`LexsiLabs_Colab_GPU.ipynb`) are open-source:
- **GitHub Repository:** `[https://github.com/sawantviraj465/LexsiLabs](https://github.com/Viraj465/Red-Team-Agents-LexsiLabs.git)`
- **Lexsi Labs Ecosystem:** `aligntune`, `curatorkit`, `safetune`
- **Benchmark:** CybORG CAGE Challenge 4
  
