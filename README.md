# SNAP: Adapting LLMs for Credit Scoring via Self-Attentive Numerical Awareness & Profiling
## Scripts
CUDA_VISIBLE_DEVICES=0 nohup python -m methods.machine_learning.model &
CUDA_VISIBLE_DEVICES=0 nohup python -m methods.informed_gpt.model &
CUDA_VISIBLE_DEVICES=0 nohup python trainer.py --experiment_name calm &
CUDA_VISIBLE_DEVICES=0 nohup python trainer.py --experiment_name snap &
CUDA_VISIBLE_DEVICES=0 nohup python trainer.py --experiment_name snap --disable_numerical_embedding &
CUDA_VISIBLE_DEVICES=0 nohup python trainer.py --experiment_name snap --disable_numerical_profiling &
CUDA_VISIBLE_DEVICES=0 nohup python trainer.py --experiment_name snap --disable_projector &

## Evaluation Setups
Area Under the Receiver Operating Characteristic Curve, Gini Coefficient, Kolmogorov–Smirnov, Precision-Recall Area Under Curve
## Research Questions
### RQ1: ablation study
Evaluate the effectiveness of each components:
w/o SNAP (pure lora)
w/o Numerical Embedding (use 23 plain embeddings to replace it)
w/o Numerical Profiling (use numerical embedding and projection)
w/o Projection (use numerical embedding and profiling)
SNAP
### RQ2: performance analysis
Traditional machien learning models, zero-shot prompting (Informed GPT), lora (CALM), and SNAP
### RQ3: visualization analysis
Visualize the ROC curves and decile analysis.
### RQ4: feature robustness (optional)
Delete feature columns under different proportion and see the performance changes (25%, 50%, 75%)
## Title Alternatives
Breaking Numerical Blindness: Intra-Numerical Prompt Tuning for Credit Risk Assessment
Overcoming Numerical Blindness of LLMs in Credit Risk Assessment
Beyond Textual Semantics: Learning Numerical Feature Interactions with LLMs for Credit Risk Assessment
The Language of Risk: Teaching LLMs to Understand Numerical Interactions in Credit Data
Closing the Gap: Enabling Large Language Models to Reason with Numerical Features in Credit Risk Assessment
