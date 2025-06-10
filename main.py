from RL import RL

# Train RL 
rl = RL(limit_dataset=10000)
rl.run_training()

# run evaluation
# rl.evaluate_model(num_samples=10000)


# save model and upload it 
repo_name = "model_enhance"
repo_id = "amira20nasser/" + repo_name
create_repo(repo_id, private=False)
upload_folder(
    repo_id=repo_id,
    folder_path="/kaggle/working/rlhf_output/final_model",  # your model folder
    path_in_repo="",
    commit_message="Initial model upload from Kaggle"
)




# to check if the parameters is changed or not 
# import copy
# import torch

# for old_param, new_param in zip(ref_model.parameters(), model.parameters()):
#     if not torch.equal(old_param.data, new_param.data):
#         print("Parameter changed.")
#         break
#     else:
#         print("No parameters changed.")
#         break
# total_change = 0.0
# for old_param, new_param in zip(ref_model.parameters(), model.parameters()):
#     total_change += torch.norm(old_param.data - new_param.data).item()
# total_change