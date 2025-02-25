import numpy as np
import torch
from peft import LoraConfig


def get_dynamic_peft_config(client_id, sample_num, data_quality=None, base_rank=8):
    """
    Dynamically configure PEFT parameters based on client data characteristics

    Args:
        client_id: ID of the client
        sample_num: Number of samples for this client
        data_quality: Optional quality score (0-1) for client data
        base_rank: Base LoRA rank for reference

    Returns:
        Configured PEFT config with adjusted rank
    """
    # Calculate sample size relative to a reference size (e.g., 1000 samples)
    relative_size = sample_num / 1000

    # Calculate adjusted rank based on sample size and quality
    if data_quality is not None:
        # Combine quantity and quality factors
        adjustment_factor = min(2.0, max(0.25, relative_size * data_quality * 2))
    else:
        # Only use quantity if quality metric not available
        adjustment_factor = min(2.0, max(0.25, relative_size))

    # Clamp the rank between min and max values
    adjusted_rank = max(2, min(32, int(base_rank * adjustment_factor)))

    # Adjust dropout based on data quantity (smaller datasets get higher dropout)
    adjusted_dropout = min(0.2, max(0.05, 0.1 + (1000 - sample_num) / 10000))

    # Log the dynamic configuration
    print(
        f"Client {client_id} with {sample_num} samples (quality: {data_quality:.4f if data_quality is not None else 'N/A'}): "
        f"using LoRA rank {adjusted_rank}, dropout {adjusted_dropout:.3f}")

    return LoraConfig(
        r=adjusted_rank,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=adjusted_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )


def assess_data_quality(dataset, tokenizer):
    """
    Evaluate the quality of a client's dataset

    Args:
        dataset: Client's dataset
        tokenizer: Tokenizer for text analysis

    Returns:
        quality_score: A score between 0-1 representing data quality
    """
    if len(dataset) == 0:
        return 0.0

    # Metric 1: Response length diversity (avoid repetitive/similar responses)
    response_lengths = []
    unique_responses = set()

    for item in dataset:
        if isinstance(item, dict) and 'response' in item:
            response = item['response']
            response_lengths.append(len(response))
            unique_responses.add(response)
        elif isinstance(item, dict) and 'text' in item:
            # Try to extract response from text
            text = item['text']
            if "assistant:" in text.lower():
                response = text.lower().split("assistant:")[-1].strip()
                response_lengths.append(len(response))
                unique_responses.add(response)

    # Handle empty or invalid datasets
    if not response_lengths:
        return 0.5  # Default moderate quality

    # Length diversity score
    length_std = np.std(response_lengths) if len(response_lengths) > 1 else 0
    length_diversity = min(1.0, length_std / 100)  # Normalize

    # Response uniqueness score
    uniqueness = len(unique_responses) / len(dataset) if len(dataset) > 0 else 0

    # Metric 2: Vocabulary richness
    all_text = ""
    for item in dataset:
        if isinstance(item, dict):
            if 'response' in item:
                all_text += " " + item['response']
            elif 'text' in item:
                all_text += " " + item['text']

    # Calculate vocabulary richness if we have text
    if all_text:
        try:
            tokens = tokenizer.tokenize(all_text)
            unique_tokens = len(set(tokens))
            vocab_richness = min(1.0, unique_tokens / 1000)  # Normalize
        except:
            vocab_richness = 0.5  # Default if tokenization fails
    else:
        vocab_richness = 0.5  # Default

    # Combine metrics (weighted average)
    quality_score = 0.4 * length_diversity + 0.3 * uniqueness + 0.3 * vocab_richness

    return quality_score


def transfer_matching_parameters(model, global_dict):
    """
    Transfer parameters from global dictionary to model where the shapes match

    Args:
        model: Local client model with potentially different architecture
        global_dict: Global model state dict
    """
    model_dict = model.state_dict()

    # Find matching parameter keys and shapes
    for key in model_dict.keys():
        if key in global_dict and model_dict[key].shape == global_dict[key].shape:
            model_dict[key] = global_dict[key].clone()

    # Load the matched parameters
    model.load_state_dict(model_dict)


def heterogeneous_model_aggregation(global_dict, local_dict_list, sample_num_list,
                                    data_quality_list, clients_this_round, quality_weighted=False):
    """
    Aggregate heterogeneous models with potentially different adapter structures

    Args:
        global_dict: Global model state dict
        local_dict_list: List of local model state dicts
        sample_num_list: Number of samples per client
        data_quality_list: Quality score per client
        clients_this_round: Clients participating in this round
        quality_weighted: Whether to weight by data quality

    Returns:
        Updated global_dict
    """
    # Group parameters by layer and adapter
    parameter_groups = {}

    # Collect all parameters from all models
    for client in clients_this_round:
        local_dict = local_dict_list[client]

        for key in local_dict.keys():
            # Extract layer and adapter info from parameter name
            parts = key.split('.')
            if 'lora' in key:
                # This is a LoRA parameter
                layer_name = '.'.join(parts[:-3])  # Get base layer name
                adapter_part = '.'.join(parts[-3:])  # Get adapter part (e.g., 'lora_A')
                group_key = (layer_name, adapter_part)

                # Initialize group if not exists
                if group_key not in parameter_groups:
                    parameter_groups[group_key] = []

                # Calculate weight based on sample size and optionally quality
                if quality_weighted and data_quality_list is not None:
                    weight = sample_num_list[client] * data_quality_list[client]
                else:
                    weight = sample_num_list[client]

                # Store parameter, weight, and shape
                parameter_groups[group_key].append({
                    'param': local_dict[key],
                    'weight': weight,
                    'client': client
                })

    # Initialize updated global dict from current global dict
    updated_global_dict = {k: v.clone() for k, v in global_dict.items()}

    # Aggregate parameters by group
    for group_key, params in parameter_groups.items():
        layer_name, adapter_part = group_key
        full_key = f"{layer_name}.{adapter_part}"

        # Skip if this key is not in global dict (may be specific to client architecture)
        if full_key not in updated_global_dict:
            continue

        # Filter params with matching shape to global param
        global_shape = updated_global_dict[full_key].shape
        matching_params = [p for p in params if p['param'].shape == global_shape]

        if not matching_params:
            continue  # No matching parameters for this key

        # Calculate total weight for normalization
        total_weight = sum(p['weight'] for p in matching_params)

        if total_weight == 0:
            continue  # Avoid division by zero

        # Weighted average of parameters
        weighted_sum = sum(p['param'] * (p['weight'] / total_weight) for p in matching_params)
        updated_global_dict[full_key] = weighted_sum

    return updated_global_dict
