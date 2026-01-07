# SNAP: Adapting LLMs for Credit Risk Assessment via Self-Attentive Numerical Alignment & Profiling
## Scripts
nohup python methods.tabllm.model &
CUDA_VISIBLE_DEVICES=0 nohup python trainer.py --experiment_name calm &
CUDA_VISIBLE_DEVICES=1 nohup python trainer.py --experiment_name snap &
CUDA_VISIBLE_DEVICES=2 nohup python trainer.py --experiment_name snap --use_numerical_embedding &
CUDA_VISIBLE_DEVICES=3 nohup python trainer.py --experiment_name snap --use_multi_head_self_attn &
## Evaluation Setups
Area Under the Curve, Kolmogorov–Smirnov
## Research Questions
### RQ1: ablation study
Evaluate the effectiveness of each components:
w/o SNAP (pure lora)
w/o Numerical Embedding (use 23 plain embeddings to replace it)
w/o Multi-Head Self-Attention (use numerical embedding and lora)
SNAP
### RQ2: performance analysis
Traditional machien learning models, zero-shot prompting (TabLLM), lora (CALM), and SNAP
### RQ3: feature robustness
Delete feature columns under different proportion and see the performance changes (25%, 50%, 75%)
## Title Alternatives
Breaking Numerical Blindness: Intra-Numerical Prompt Tuning for Credit Risk Assessment
Overcoming Numerical Blindness of LLMs in Credit Risk Assessment
Beyond Textual Semantics: Learning Numerical Feature Interactions with LLMs for Credit Risk Assessment
The Language of Risk: Teaching LLMs to Understand Numerical Interactions in Credit Data
Closing the Gap: Enabling Large Language Models to Reason with Numerical Features in Credit Risk Assessment
