import math


def get_proxy_dict(fed_args, global_dict):
    proxy_dict = {}
    opt_proxy_dict = {}

    for key in global_dict.keys():
        proxy_dict[key] = torch.zeros_like(global_dict[key])
        opt_proxy_dict[key] = torch.zeros_like(global_dict[key])

    return proxy_dict, opt_proxy_dict


def get_auxiliary_dict(fed_args, global_dict):
    global_auxiliary = {}
    auxiliary_model_list = []
    auxiliary_delta_dict = []

    if fed_args.fed_alg == 'scaffold':
        for key in global_dict.keys():
            global_auxiliary[key] = torch.zeros_like(global_dict[key])

        for i in range(fed_args.num_clients):
            auxiliary_model_list.append(copy.deepcopy(global_auxiliary))
            auxiliary_delta_dict.append(copy.deepcopy(global_auxiliary))

    return global_auxiliary, auxiliary_model_list, auxiliary_delta_dict


def get_dataset_this_round(dataset, round, fed_args, script_args):
    """
    Get the required sub-dataset for this round
    """
    if fed_args.fed_alg.startswith('local'):
        step_size = len(dataset) // fed_args.num_rounds
        if step_size == 0:  # dataset too small
            return dataset

        start_idx = round * step_size
        end_idx = start_idx + step_size if round < fed_args.num_rounds - 1 else len(dataset)

        return dataset.select(range(start_idx, end_idx))
    else:
        return dataset


def cosine_learning_rate(current_step, total_steps, init_lr, min_lr):
    """
    Calculate cosine learning rate decay
    """
    progress = current_step / total_steps
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (init_lr - min_lr) * cosine_decay


def cosine_learning_rate(current_round, total_rounds, initial_lr=0.001, min_lr=0):
    """
    Compute the learning rate based on a cosine schedule.

    :param current_round: The current training round (0-indexed).
    :param total_rounds: The total number of training rounds.
    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    :return: The computed learning rate for the current round.
    """
    # Compute the cosine learning rate
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * current_round / total_rounds))
    return cosine_lr


def get_dynamic_local_step(local_dataset_length, script_args, fed_args, epoches=3):
    sample_num = fed_args.num_rounds * fed_args.sample_clients / fed_args.num_clients
    sample_data_num = local_dataset_length * epoches / sample_num
    local_step = math.ceil(sample_data_num / (script_args.batch_size * script_args.gradient_accumulation_steps))
    return local_step


if __name__ == "__main__":

    # Example usage:
    num_rounds = 300
    initial_lr = 5e-5
    min_lr = 1e-6

    lr_list = []
    for round in range(num_rounds):
        lr = cosine_learning_rate(round, num_rounds, initial_lr, min_lr)
        lr_list.append(lr)
        print(f"Round {round + 1}/{num_rounds}, Learning Rate: {lr:.8f}")
