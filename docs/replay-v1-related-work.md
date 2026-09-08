# REPLAY-V1: closest related work

Checked 2026-09-05. This is a bounded positioning note, not a novelty verdict or
design amendment. The five specified papers were checked beyond their abstracts,
including their experimental appendices. The text versions below are explicit;
conference status is verified separately, not a claim that an arXiv version and
the proceedings PDF are byte-identical.

## Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting

Howard Chen, Noam Razin, Karthik Narasimhan, and Danqi Chen.
arXiv:2510.18874v3, 26 June 2026; first posted 21 October 2025.
ICML 2026, as recorded in the [author-submitted bibliographic record](https://arxiv.org/abs/2510.18874v3).

The experiments compare SFT, self-generated SFT, and GRPO on instruction
following, knowledge, and arithmetic using Llama/Qwen models. They measure
pre-existing capabilities while acquiring the target task. Approximately
on-policy data reduces forgetting; KL regularization and advantage estimation
do not alone explain the difference. Importantly, Appendix A.4.1/Figure 10
already trains SFT on traces collected during RL, with substantially reduced
forgetting. Those traces are not on-policy for the SFT learner. See
[full text, §§2–4 and Appendices A.3–A.5](https://arxiv.org/html/2510.18874v3).

**Overlap and limit.** SFT-on-RL-traces is existing work, not a REPLAY-V1
invention. The distinction being tested here is the acquired skill's subsequent
durability and later-task learning after a common follow-up training procedure.
The paper's mixture-model analysis is intuition under specific assumptions,
not a general guarantee about neural-model retention.

## RL's Razor: Why Online Reinforcement Learning Forgets Less

Idan Shenfeld, Jyothish Pari, and Pulkit Agrawal.
arXiv:2509.04259v1, 4 September 2025; subsequently
[ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/618c95f4557c15b253fb0e6f548ea0c0-Abstract-Conference.html).

Qwen2.5-3B-Instruct and OpenVLA experiments compare new-task/prior-task
trade-offs across hyperparameter sweeps. On-policy GRPO and correct-only
REINFORCE retain more prior performance than offline SFT/SimPO. Forward KL
on new-task inputs predicts forgetting in their experiments. The toy
ParityMNIST study includes both an oracle SFT distribution and SFT distillation
from an RL teacher; the latter matches the teacher's trade-off within noise.
See [full text, §§3–6, Figure 9, Appendices A–C](https://arxiv.org/html/2509.04259v1).

**Overlap and limit.** Both data-distribution explanations and RL-teacher
distillation predate REPLAY-V1. Their primary comparison is retention during
SFT versus RL acquisition, not durability of separately acquired skills under
the same later training. Appendix A explicitly limits its projection theorem:
neural-network policy families need not satisfy the required geometry. Its
empirical KL relationship is not proof that a LoRA factor norm causes
forgetting, nor that a later-task continuation must preserve an acquisition-time
advantage.

## SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training

Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie,
Dale Schuurmans, Quoc V. Le, Sergey Levine, and Yi Ma.
arXiv:2501.17161v2, 26 May 2025; first posted 28 January 2025.
[ICML 2025, PMLR 267:10818–10838](https://proceedings.mlr.press/v267/chu25c.html).

Using Llama-3.2-Vision-11B, the study compares further SFT with multi-turn PPO
after a shared SFT initialization. GeneralPoints arithmetic and V-IRL navigation
provide textual-rule and visual distribution shifts. RL improves the tested OOD
measures where SFT often degrades them. SFT initialization enables valid output;
extreme SFT overfitting can leave later RL unable to restore OOD performance.
Appendix C.1 also tests SFT with suboptimal, revision-bearing trajectories.
See [full text, §§3–6 and Appendices C–D](https://arxiv.org/html/2501.17161v2).

**Overlap and limit.** It establishes acquisition/generalization and
initialization-quality questions, not a universal theorem that SFT memorizes.
Its multi-turn verifier-assisted success and navigation per-step accuracy are
not interchangeable with DuraSeed's raw single-completion rates. REPLAY-V1
changes the fixed trace corpus under a common supervised acquisition recipe;
it does not reproduce this paper's PPO-versus-SFT comparison or isolate
reasoning from all response-style differences.

## Good SFT Optimizes for SFT, Better SFT Prepares for Reinforcement Learning

Dylan Zhang, Yufeng Xu, Haojin Wang, Qingzhi Chen, and Hao Peng.
arXiv:2602.01058v2, 28 May 2026; first posted 1 February 2026.
[ICML 2026 poster](https://icml.cc/virtual/2026/poster/62456).

The paper varies offline SFT objectives, then applies the same later GRPO
recipe. Offline score rankings can reverse after RL. PEAR weights offline
losses by behavior/target likelihood ratios, including continuation-sensitive
weights, and improves downstream results across Qwen and distilled models.
Section 4.4/Table 4 explicitly includes shifted-task continuation: offline
SynLogic training followed by Enigmata RL under a shared recipe and rollout
budget. See [full text, §§2–4 and Appendices H–M](https://arxiv.org/html/2602.01058v2).

**Overlap and limit.** The general idea that earlier training affects later
learnability—including under a different task distribution—is already studied.
DuraSeed's narrower comparison jointly measures loss of the acquired skill and
learning of the subsequent task under shared later SFT. REPLAY-V1 is not PEAR:
it uses unweighted supervised learning on two fixed, verified trace sources.
The paper's likelihood-ratio and gradient/spectral diagnostics do not establish
DuraSeed's proposed factor-scale mechanism. Performance numbers must come from
the cited v2 text rather than its older abstract summary.

## LoRA vs Full Fine-tuning: An Illusion of Equivalence

Reece Shuttleworth, Jacob Andreas, Antonio Torralba, and Pratyusha Sharma.
arXiv:2410.21228v3, 22 October 2025; first posted 28 October 2024.
[NeurIPS 2025, main conference](https://proceedings.nips.cc/paper_files/paper/2025/hash/ff541950d1e885af90f523571564a401-Abstract-Conference.html).

RoBERTa experiments and LLaMA-family checkpoints show that comparable task
performance need not imply equivalent spectral changes or forgetting for LoRA
and full fine-tuning. The analysis identifies prominent weight directions
weakly aligned with pretrained singular vectors. Rescaling these directions
can improve pretrained-distribution loss with limited task-performance change.
Sequential experiments merge/reinitialize adapters between tasks; LoRA can
accumulate these directions and forget more. See
[full text, §§3–5 and Appendices A.4–A.5, B, H, L–O](https://arxiv.org/html/2410.21228v3).

**Overlap and limit.** The spectral object is the merged weight matrix relative
to pretrained weights—not simply the concentration of BA or the magnitude of
B. The paper explicitly finds effective rank alone insufficient. DuraSeed has
no full-fine-tuning arm, so cannot establish a LoRA-versus-full-fine-tuning
effect. The current paper does not support a universal claim that LoRA forgets
more: learning rate, scaling, and sequential-training conditions matter.

## Scope of the present comparison

REPLAY-V1 asks whether assigning either of two archived, verified trace corpora
under the same supervised acquisition recipe changes the measured durability
and later-task learning of the selected checkpoints. The prompt populations and
per-update order are shared, but response lengths, trained tokens, and selected
acquisition doses need not be. This is a fixed-corpus, two-source-block follow-up
to known Pilot outcomes—not an isolated comparison of SFT and RL objectives,
not an independent estimate of training-seed variance, and not a causal test of
LoRA factor scale. None of the five papers supplies authority to change this
experiment or supports a claim of being first.
