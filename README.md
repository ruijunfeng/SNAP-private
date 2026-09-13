# SNAP: Adapting LLMs for Credit Scoring via Self-Attentive Numerical Awareness & Profiling
## Scripts
```bash
CUDA_VISIBLE_DEVICES=0 nohup python -m methods.machine_learning.model --dataset_name heloc > logs/machine_learning.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m methods.informed_gpt.model --dataset_name heloc > logs/informed_gpt.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python trainer.py --experiment_name ft_transformer --dataset_name heloc > logs/ft_transformer.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python trainer.py --experiment_name calm --dataset_name heloc > logs/calm.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python trainer.py --experiment_name snap --dataset_name heloc > logs/snap.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python trainer.py --experiment_name snap --disable_numerical_embedding --dataset_name heloc > logs/snap_disable_numerical_embedding.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python trainer.py --experiment_name snap --disable_numerical_profiling --dataset_name heloc > logs/snap_disable_numerical_profiling.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python trainer.py --experiment_name snap --disable_projector --dataset_name heloc > logs/snap_disable_projector.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python trainer.py --experiment_name snap  --disable_language_model --dataset_name heloc > logs/snap_disable_language_model.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -m methods.machine_learning.model --dataset_name pakdd > logs/machine_learning.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m methods.informed_gpt.model --dataset_name pakdd > logs/informed_gpt.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python trainer.py --experiment_name ft_transformer --dataset_name pakdd > logs/ft_transformer.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python trainer.py --experiment_name calm --dataset_name pakdd > logs/calm.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python trainer.py --experiment_name snap --dataset_name pakdd > logs/snap.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python trainer.py --experiment_name snap --disable_numerical_embedding --dataset_name pakdd > logs/snap_disable_numerical_embedding.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python trainer.py --experiment_name snap --disable_numerical_profiling --dataset_name pakdd > logs/snap_disable_numerical_profiling.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python trainer.py --experiment_name snap --disable_projector --dataset_name pakdd > logs/snap_disable_projector.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python trainer.py --experiment_name snap  --disable_language_model --dataset_name pakdd > logs/snap_disable_language_model.log 2>&1 &

GPU=0; RUNS=5; nohup sh -c "
export CUDA_VISIBLE_DEVICES=$GPU
while [ -d /proc/1799384 ]; do sleep 60; done || exit 1

python trainer.py --dataset_name pakdd --experiment_name snap || exit 1
python trainer.py --dataset_name pakdd --experiment_name snap || exit 1
python trainer.py --dataset_name pakdd --experiment_name calm || exit 1
python trainer.py --dataset_name pakdd --experiment_name calm || exit 1

for dataset_name in pakdd heloc; do
    for run in \$(seq 1 $RUNS); do
        python trainer.py --dataset_name \"\$dataset_name\" --experiment_name snap || exit 1
        python trainer.py --dataset_name \"\$dataset_name\" --experiment_name snap --disable_projector || exit 1
        if [ \$run -le 3 ]; then
            python trainer.py --dataset_name \"\$dataset_name\" --experiment_name snap --disable_numerical_embedding || exit 1
        fi
    done
done

" > "logs/experiment_${GPU}.log" 2>&1 &

GPU=0; RUNS=5; nohup sh -c "
export CUDA_VISIBLE_DEVICES=$GPU
while [ -d /proc/1799384 ]; do sleep 60; done || exit 1

python trainer.py --dataset_name pakdd --experiment_name snap || exit 1
python trainer.py --dataset_name pakdd --experiment_name snap || exit 1
python trainer.py --dataset_name pakdd --experiment_name calm || exit 1
python trainer.py --dataset_name pakdd --experiment_name calm || exit 1

for dataset_name in pakdd heloc; do
    for run in \$(seq 1 $RUNS); do
        python trainer.py --dataset_name \"\$dataset_name\" --experiment_name calm || exit 1
        python trainer.py --dataset_name \"\$dataset_name\" --experiment_name snap --disable_numerical_profiling || exit 1
        if [ \$run -le 3 ]; then
            python trainer.py --dataset_name \"\$dataset_name\" --experiment_name snap --disable_numerical_embedding || exit 1
        fi
    done
done

" > "logs/experiment_${GPU}.log" 2>&1 &
```
## Evaluation Setups
Area Under the Receiver Operating Characteristic Curve, Gini Coefficient, Kolmogorov–Smirnov, Precision-Recall Area Under Curve
## Research Questions
### RQ1: ablation study
Evaluate the effectiveness of each components:
w/o SNAP (pure lora)
w/o Numerical Embedding (use linear projection to replace it)
w/o Numerical Profiling (use numerical embedding and projection)
w/o Projection (use numerical embedding and profiling)
w/o Language Model (use projection to project to logits)
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
