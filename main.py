from RL import RL

# Train RL 
rl = RL(limit_dataset=10000)
rl.run_training()

# run evaluation
rl.evaluate_model(num_samples=10000)