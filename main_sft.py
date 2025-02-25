import copy
import os
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate, script_args.max_steps)
save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Load the dataset =====
if fed_args.fed_alg.startswith('local'):
    dataset = get_local_datasets(script_args.local_data_dir)
else:
    dataset = get_fed_datasets(script_args.local_data_dir)

# ===== Split the dataset into clients =====
local_datasets = dataset[:fed_args.num_clients]
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]
print(sample_num_list)
print(len(sample_num_list))

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=training_args.gradient_checkpointing
    )

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right",
                                          model_max_length=script_args.seq_length)
if script_args.multi_turn_task:
    tokenizer.pad_token = tokenizer.unk_token  # following vicuna
    model.resize_token_embeddings(len(tokenizer))
else:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token  # following vicuna

# ===== Assess data quality for each client =====
data_quality_list = []
for client_id in range(fed_args.num_clients):
    quality = assess_data_quality(local_datasets[client_id], tokenizer)
    data_quality_list.append(quality)
    print(f"Client {client_id} data quality score: {quality:.4f}")

# ===== Get base PEFT model =====
base_model = model  # Save reference to base model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token,
                                                                         script_args)
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[
                        2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

if script_args.multi_turn_task:
    data_collator_list, sample_num_list = get_multi_turn_dataset(fed_args.fed_alg, script_args.local_data_dir,
                                                                 tokenizer)
    print("multi-turn")
    print(len(sample_num_list))
    print(sample_num_list)

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]
client_peft_configs = {}  # Store client-specific PEFT configs

for round in tqdm(range(fed_args.num_rounds)):

    clients_this_round = get_clients_this_round(fed_args, round)

    print(f">> ==================== Round {round + 1} : {clients_this_round} ====================")

    # Store training results for this round
    round_models = {}

    for client in range(fed_args.num_clients):

        if client not in clients_this_round:
            training_loss[client].append(-1)  # -1 is an indicator of not training
            continue

        # Create dynamic PEFT config based on client data characteristics
        if script_args.dynamic_peft:
            if client not in client_peft_configs:
                # Create a new dynamic config for this client
                client_peft_configs[client] = get_dynamic_peft_config(
                    client_id=client,
                    sample_num=sample_num_list[client],
                    data_quality=data_quality_list[client],
                    base_rank=peft_config.r  # Use the original rank as base
                )

            # Reset the model to base model and apply client-specific config
            model = get_peft_model(base_model, client_peft_configs[client])

            # If this is the first time for this client, initialize with global weights
            # for layers that match between configs
            if round == 0 or client not in round_models:
                try:
                    # Transfer matching parameters from global model
                    transfer_matching_parameters(model, global_dict)
                except Exception as e:
                    print(f"Warning: Could not transfer parameters for client {client}: {e}")
        else:
            # Standard approach: use global model with original PEFT config
            set_peft_model_state_dict(model, global_dict)  # sync the global model to the local model

        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args,
                                             script_args)  # get the required sub-dataset for this round
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate,
                                      1e-6)  # manually schedule the learning rate
        training_args = get_training_args(script_args, new_lr, script_args.max_steps)

        # ===== Train local model on the client side =====
        if script_args.multi_turn_task:
            client_data_collator = data_collator_list[client]
            print('client', client, 'data number', len(client_data_collator))
            trainer = get_fed_local_sft_trainer(
                model=model,
                tokenizer=tokenizer,
                training_args=training_args,
                local_dataset=sub_dataset,
                formatting_prompts_func=formatting_prompts_func,
                data_collator=client_data_collator,
                global_dict=global_dict,
                fed_args=fed_args,
                script_args=script_args,
                local_auxiliary=auxiliary_model_list[client],
                global_auxiliary=global_auxiliary,
            )
        else:
            trainer = get_fed_local_sft_trainer(
                model=model,
                tokenizer=tokenizer,
                training_args=training_args,
                local_dataset=sub_dataset,
                formatting_prompts_func=formatting_prompts_func,
                data_collator=data_collator,
                global_dict=global_dict,
                fed_args=fed_args,
                script_args=script_args,
                local_auxiliary=auxiliary_model_list[client],
                global_auxiliary=global_auxiliary,
            )

        results = trainer.train()
        training_loss[client].append(results.training_loss)

        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

        local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))  # copy is needed!
        round_models[client] = model  # Save reference to trained model

    # ===== Server aggregates the local models =====
    if script_args.dynamic_peft:
        # For dynamic PEFT, we need special handling for different adapter structures
        global_dict = heterogeneous_model_aggregation(
            global_dict, local_dict_list, sample_num_list, data_quality_list,
            clients_this_round, script_args.quality_weighted_aggregation
        )
    else:
        # Standard aggregation for homogeneous models
        global_dict, global_auxiliary = global_aggregate(
            fed_args, script_args, global_dict, local_dict_list, sample_num_list,
            clients_this_round, round, proxy_dict=proxy_dict,
            opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict),
            data_quality_list=data_quality_list if script_args.quality_weighted_aggregation else None
        )

    # Restore the model to standard PEFT config for global model
    model = get_peft_model(base_model, peft_config)
    set_peft_model_state_dict(model, global_dict)  # Update global model

    # ===== Save the model =====
    if (round + 1) % fed_args.checkpoint_step == 0:
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round + 1}"))

    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))