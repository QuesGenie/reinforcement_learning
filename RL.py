import os
import torch
import gc
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
# trl version is 0.11.3
from trl import PPOConfig, PPOTrainer, AutoModelForSeq2SeqLMWithValueHead
from sentence_transformers import CrossEncoder

class RL:
    def __init__(
        self, 
        model_name="fares7elsadek/t5-base-finetuned-question-generation",
        reward_model_name="cross-encoder/qnli-distilroberta-base",
        dataset_name="squad",
        output_dir="./rlhf_output",
        batch_size=4,
        mini_batch_size=2,
        learning_rate=1.41e-5,
        max_seq_length=64,
        max_new_tokens=64,
        checkpoint_interval=100,
        evaluation_samples=50,
        log_interval=10,
        limit_dataset=10000,
    ):
        self.model_name = model_name
        self.reward_model_name = reward_model_name
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_interval = checkpoint_interval
        self.evaluation_samples = evaluation_samples
        self.log_interval = log_interval
        self.limit_dataset = limit_dataset
        self.max_seq_length = max_seq_length

        os.makedirs(output_dir, exist_ok=True)

        self.ppo_config = PPOConfig(
            model_name=model_name,
            learning_rate=learning_rate,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            gradient_accumulation_steps=2,
            ppo_epochs=4,
            optimize_cuda_cache=True,
            # clip_range=0.2
        )
        self.ppo_config.clip_range = 0.2  
        self.generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_k": 0,
            "top_p": 0.95,
            "temperature": 1.0,
            "use_cache": True,
        }
        
        self.stats = {
            "epoch": [],
            "avg_reward": [],
            "policy_loss": [],
            "value_loss": [],
            "elapsed_time": [],
        }
        
        self.checkpoint_stats = []

    def print_gpu_memory(self):
            """Print GPU memory usage stats."""
            if torch.cuda.is_available():
                print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB allocated, {torch.cuda.memory_reserved() / 1024**2:.2f}MB reserved")
                
    def clear_memory(self):
        """Clear memory to avoid OOM errors."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def load_models(self):
        """Load and prepare all models."""
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Loading models...")
        self.model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(self.model_name)
        self.ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(self.model_name)
        
        print("Loading reward model...")
        self.reward_model = CrossEncoder(self.reward_model_name)
        
        # self.generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        # self.generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        print("All models loaded successfully")
        
    def load_data(self):
        """Load and prepare dataset."""
        print(f"Loading {self.dataset_name} dataset (limit: {self.limit_dataset})...")
        dataset = load_dataset(self.dataset_name)
        self.train_dataset = dataset['train'].shuffle(seed=42).select(range(self.limit_dataset))
        self.train_dataset = self.train_dataset.map(self.tokenize_sample, batched=False)
        print(f"Dataset prepared with {len(self.train_dataset)} samples")
        
    def tokenize_sample(self, sample):
        """Tokenize a single sample."""
        context = sample["context"]
        tokens = self.tokenizer(
            context, 
            padding=False, 
            truncation=True, 
            max_length=self.max_seq_length
        )
        sample["input_ids"] = tokens["input_ids"]
        sample["attention_mask"] = tokens["attention_mask"]
        sample["query"] = self.tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)
        return sample
        
    def collate_fn(self, batch):
        """Collate function for dataloader."""
        input_ids = [torch.tensor(sample["input_ids"], dtype=torch.long) for sample in batch]
        attention_masks = [torch.tensor(sample["attention_mask"], dtype=torch.long) for sample in batch]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, 
            batch_first=True, 
            padding_value=0
        )
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "query": [sample["query"] for sample in batch],
        }
    
    def init_trainer(self):
        """Initialize PPO trainer."""
        print("Initializing PPO trainer...")
        self.ppo_trainer = PPOTrainer(
            self.ppo_config,
            self.model,
            self.ref_model,
            self.tokenizer,
            dataset=self.train_dataset,
            data_collator=self.collate_fn
        )
        
        self.train_device = self.ppo_trainer.accelerator.device
        print(f"PPO Trainer using device: {self.train_device}")
        self.ppo_trainer.model = self.ppo_trainer.model.to(self.train_device)
        
    def calculate_reward(self, contexts, answers, questions):
        """Calculate reward scores using the reward model."""
        input_pairs = [(questions[i] + " " + answers[i], contexts[i]) for i in range(len(questions))]
        with torch.no_grad():
            scores = self.reward_model.predict(input_pairs)
        return [torch.tensor(score, device=self.train_device) for score in scores]
    
    def save_checkpoint(self, epoch, avg_reward=None):
        """Save model checkpoint with metadata."""
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model
        self.ppo_trainer.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save metadata
        metadata = {
            "epoch": epoch,
            "avg_reward": avg_reward,
            "timestamp": time.strftime("%Y-%m-%d-%H-%M-%S"),
            # "gpu_memory": self.print_gpu_memory()
        }
        
        self.checkpoint_stats.append(metadata)
        
        # Save stats as CSV
        stats_df = pd.DataFrame(self.checkpoint_stats)
        stats_df.to_csv(os.path.join(self.output_dir, "checkpoint_stats.csv"), index=False)
        
        print(f"Checkpoint saved at epoch {epoch} with avg reward: {avg_reward:.4f}")
    
    def run_training(self, num_epochs=None):
        """Run the full training loop with automatic checkpointing"""
        self.load_models()
        self.load_data()
        self.init_trainer()
        
        print("Starting training...")
        start_time = time.time()
        
        epoch = 0
        total_steps = 0
        
        # If num_epochs is None, use the full dataloader
        max_epochs = num_epochs if num_epochs is not None else len(self.ppo_trainer.dataloader)
        
        progress_bar = tqdm(total=max_epochs, desc="Training Progress")
        
        for epoch, batch in enumerate(self.ppo_trainer.dataloader):
            if num_epochs is not None and epoch >= num_epochs:
                break
                
            epoch_start_time = time.time()
            epoch_rewards = []
            
            try:
                print(f"\n======= Epoch {epoch} =======")
                self.print_gpu_memory()
                # mem_info = self.print_gpu_memory()
                current_device = self.ppo_trainer.accelerator.device
                
                # Move batch to device
                input_ids = batch["input_ids"].to(current_device)
                attention_mask = batch["attention_mask"].to(current_device)
                queries = batch["query"]
                
                query_tensors = []
                response_tensors = []
                
                # Generate responses
                for i in range(len(input_ids)):
                    sample_input = input_ids[i:i+1].to(current_device)
                    sample_mask = attention_mask[i:i+1].to(current_device)
                    
                    query_tensors.append(sample_input.squeeze(0).detach().clone())
                    
                    output = self.ppo_trainer.model.generate(
                        sample_input,
                        attention_mask=sample_mask,
                        **self.generation_kwargs
                    )
                    
                    response_tensors.append(output[0].detach().clone().to(current_device))
                
                # Decode responses
                decoded_responses = [
                    self.tokenizer.decode(output, skip_special_tokens=True)
                    for output in response_tensors
                ]
                
                # Extract questions and answers
                batch['response'] = decoded_responses
                batch["response_q"] = [r.split("answer:")[0].strip().replace("question: ", "").strip() 
                                      for r in decoded_responses]
                batch["response_ans"] = [r.split("answer:")[1].strip() if "answer:" in r else "" 
                                        for r in decoded_responses]
                
                # Calculate rewards
                rewards = self.calculate_reward(queries, batch["response_ans"], batch["response_q"])
                epoch_rewards.extend([r.item() for r in rewards])
                
                # Ensure tensors are on the correct device
                for i in range(len(query_tensors)):
                    if query_tensors[i].device != current_device:
                        query_tensors[i] = query_tensors[i].to(current_device)
                    if response_tensors[i].device != current_device:
                        response_tensors[i] = response_tensors[i].to(current_device)
                    if rewards[i].device != current_device:
                        rewards[i] = rewards[i].to(current_device)
                
                # Run PPO step
                stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
                self.ppo_trainer.log_stats(stats, batch, rewards)
                
                # Log statistics
                avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
                
                # Track stats
                self.stats["epoch"].append(epoch)
                self.stats["avg_reward"].append(avg_reward)
                self.stats["policy_loss"].append(stats["ppo/loss/policy"])
                self.stats["value_loss"].append(stats["ppo/loss/value"])
                self.stats["elapsed_time"].append(time.time() - start_time)
                
                # Log progress
                if epoch % self.log_interval == 0:
                    print(f"Epoch: {epoch}, Avg Reward: {avg_reward:.4f}, "
                          f"Policy Loss: {stats['ppo/loss/policy']:.4f}, "
                          f"Value Loss: {stats['ppo/loss/value']:.4f}")
                
                # Save checkpoint
                if epoch % self.checkpoint_interval == 0 and epoch > 0:
                    self.save_checkpoint(epoch, avg_reward)
                
                del input_ids, attention_mask, query_tensors, response_tensors
                self.clear_memory()
                
                total_steps += 1
                progress_bar.update(1)
                
            except Exception as e:
                print(f"Error in epoch {epoch}: {str(e)}")
                print(f"Model device: {next(self.ppo_trainer.model.parameters()).device}")
                print(f"Ref model device: {next(self.ppo_trainer.ref_model.parameters()).device}")
                self.print_gpu_memory()
                raise e
        
        progress_bar.close()
        
        final_checkpoint_path = os.path.join(self.output_dir, "final_model")
        self.ppo_trainer.model.save_pretrained(final_checkpoint_path)
        self.tokenizer.save_pretrained(final_checkpoint_path)
        
        self.save_training_stats()
        
        # evaluation
        # print("Running final evaluation...")
        # self.evaluate_model()
        
        print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes")
        
    def save_training_stats(self):
        """Save training statistics and generate plots."""
        # Save stats as CSV
        stats_df = pd.DataFrame(self.stats)
        stats_df.to_csv(os.path.join(self.output_dir, "training_stats.csv"), index=False)
        
    def evaluate_model(self, num_samples=None):
        """Evaluate the model against the reference model."""
        print("Evaluating model...")
        
        if num_samples is None:
            num_samples = self.evaluation_samples
            
        # Make sure we don't exceed dataset size
        num_samples = min(num_samples, len(self.train_dataset))
        
        self.ref_model.eval()
        self.model.eval()
        
        results = []
        total_improvement = 0
        improvement_count = 0
        best_improvements = []
        
        with torch.no_grad():
            for i in tqdm(range(num_samples), desc="Evaluation"):
                context = self.train_dataset[i]["context"]
                inputs = self.tokenizer(context, return_tensors="pt").input_ids.to(self.train_device)
                
                # Generate with reference model
                ref_outputs = self.ref_model.generate(inputs, **self.generation_kwargs)
                ref_question = self.tokenizer.decode(ref_outputs[0], skip_special_tokens=True)
                
                # Generate with trained model
                trained_outputs = self.model.generate(inputs, **self.generation_kwargs)
                trained_question = self.tokenizer.decode(trained_outputs[0], skip_special_tokens=True)
                
                # Extract question and answer parts
                ref_q = ref_question.split("answer:")[0].strip().replace("question: ", "").strip()
                trained_q = trained_question.split("answer:")[0].strip().replace("question: ", "").strip()
                
                ref_ans = ref_question.split("answer:")[1].strip() if "answer:" in ref_question else ""
                trained_ans = trained_question.split("answer:")[1].strip() if "answer:" in trained_question else ""
                
                # Calculate scores
                score_ref = self.reward_model.predict((ref_q + " " + ref_ans, context))
                score_trained = self.reward_model.predict((trained_q + " " + trained_ans, context))
                
                improvement = score_trained - score_ref
                
                # Store result
                results.append({
                    "context": context,
                    "ref_question": ref_q + " answer: " + ref_ans,
                    "trained_question": trained_q + " answer: " + trained_ans,
                    "ref_score": score_ref,
                    "trained_score": score_trained,
                    "improvement": improvement
                })
                
                total_improvement += improvement
                if improvement > 0:
                    improvement_count += 1
                
                if improvement > 0.5:
                    best_improvements.append(results[-1])
        
        # Calculate statistics
        average_improvement = total_improvement / num_samples
        
        # Print results
        print(f"Evaluation Results:")
        print(f"Total samples evaluated: {num_samples}")
        print(f"Samples with improvement: {improvement_count} ({(improvement_count/num_samples)*100:.2f}%)")
        print(f"Average improvement: {average_improvement:.4f}")
        
        # Show some best examples
        print("\nBest Improvement Examples:")
        for i, example in enumerate(sorted(best_improvements, key=lambda x: x['improvement'], reverse=True)[:5]):
            print(f"\nExample {i+1}:")
            print(f"Context: {example['context'][:300]}...")
            print(f"Reference: {example['ref_question']}")
            print(f"Improved: {example['trained_question']}")
            print(f"Improvement: {example['improvement']:.4f}")
            print("-" * 50)
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.output_dir, "evaluation_results.csv"), index=False)
        
        # Save best improvements
        if best_improvements:
            best_df = pd.DataFrame(best_improvements)
            best_df.to_csv(os.path.join(self.output_dir, "best_improvements.csv"), index=False)
            
        # Create pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        labels = ['Improved', 'Not Improved']
        sizes = [improvement_count, num_samples - improvement_count]
        colors = ['green', 'lightcoral']
        explode = (0.1, 0)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90)
        ax.set_title('Model Improvement Distribution')
        
        plt.savefig(os.path.join(self.output_dir, "improvement_distribution.png"))
        
        return {
            "num_samples": num_samples,
            "improvement_count": improvement_count,
            "average_improvement": average_improvement,
            "best_improvements": best_improvements[:5] if best_improvements else []
        }
