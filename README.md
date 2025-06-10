# Reinforcement Learning with PPO for Question Generation

This project provides a Python class RL for applying Reinforcement Learning from Human Feedback (RLHF) to improve question generation using Proximal Policy Optimization (PPO). It uses Hugging Face Transformers, trl, and SentenceTransformers to fine-tune a Seq2Seq model like T5 for generating better questions based on context, guided by a reward model.


## Dependencies

Ensure you have the following libraries installed:

```bash
pip install torch transformers datasets sentence-transformers trl tqdm matplotlib pandas
```
> **_NOTE:_** The version of trl used: 0.11.3

## Quick Start

### 1. Initialize the Class

```python
from RL import RL
rl_trainer = RL()
```

You can also pass custom configurations:

```python
rl_trainer = RL(
    model_name="fares7elsadek/t5-base-finetuned-question-generation",
    reward_model_name="cross-encoder/qnli-distilroberta-base",
    dataset_name="squad",
    output_dir="./rlhf_output",
)-
```
### 2. Run Training

```python
rl_trainer.run_training()
```
This method:
- Loads models and dataset
- Initializes PPO trainer
- Runs PPO training 

# Reward Model
The reward model used is a CrossEncoder, which scores how well the generated Q&A pair aligns with the context. Default is: 

[link reward model](https://huggingface.co/cross-encoder/qnli-distilroberta-base)

```python
reward_model = CrossEncoder(cross-encoder/qnli-distilroberta-base)
```





