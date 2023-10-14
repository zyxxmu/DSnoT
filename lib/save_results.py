import os

def save_ppl_result(args, output_file, sparsity_ratio, ppl):
    if os.path.exists(output_file):
        open_mode = "a"
    else:
        open_mode = "w"
    
    with open(output_file, open_mode) as f:
        f.write(f"model: {args.model}\n")
        f.write(f"prune_method: {args.prune_method}\n")
        f.write(f"without_DSnoT: {args.without_DSnoT}\n")
        f.write(f"initial_method: {args.initial_method}\n")
        f.write(f"skip_layer {args.skip_layer}, skip_sub_layer {args.skip_sub_layer}\n")
        f.write(f"max_cycle_time: {args.max_cycle_time}, update_threshold: {args.update_threshold}\n")
        f.write(f"pow_of_var_pruning: {args.pow_of_var_pruning}, pow_of_var_regrowing:{args.pow_of_var_regrowing}\n")
        f.write(f"without_same_sign: {args.without_same_sign}\n")
        f.write(f"sparse pattern: {args.sparsity_type}\n")
        f.write(f"sample: {args.nsamples}\n")
        f.write(f"sparsity sanity check {sparsity_ratio:.4f}, ppl: {ppl}\n\n")

    
    # print same info to terminal
    print(f"model: {args.model}")
    print(f"prune_method: {args.prune_method}")
    print(f"without_DSnoT: {args.without_DSnoT}")
    print(f"initial_method: {args.initial_method}")
    print(f"skip_layer {args.skip_layer}, skip_sub_layer {args.skip_sub_layer}")
    print(f"max_cycle_time: {args.max_cycle_time}, update_threshold: {args.update_threshold}")
    print(f"pow_of_var_pruning:{args.pow_of_var_pruning}, pow_of_var_regrowing:{args.pow_of_var_regrowing}")
    print(f"without_same_sign:{args.without_same_sign}")
    print(f"sparse pattern: {args.sparsity_type}")
    print(f"sample: {args.nsamples}")
    print(f"sparsity sanity check {sparsity_ratio:.4f}, ppl: {ppl}\n\n")


def save_time_result(args, output_file, total_time):
    if os.path.exists(output_file):
        open_mode = "a"
    else:
        open_mode = "w"
    
    with open(output_file, open_mode) as f:
        f.write(f"model: {args.model}\n")

        if args.prune_method == "DSnoT":
            f.write(f"prune method: {args.prune_method}, without_DSnoT: {args.without_DSnoT}\n")
            f.write(f"initial_method: {args.initial_method}\n")
            f.write(f"skip_layer {args.skip_layer}, skip_sub_layer {args.skip_sub_layer}\n")
        else:
            f.write(f"prune method: {args.prune_method}\n")
        f.write(f"sparsity_ratio: {args.sparsity_ratio}\n")
        f.write(f"total_time: {total_time}\n\n")
    
    # print same info to terminal
    print(f"model: {args.model}")
    if args.prune_method == "DSnoT":
        print(f"prune method: {args.prune_method}, without_DSnoT: {args.without_DSnoT}")
        print(f"initial_method: {args.initial_method}")
        print(f"skip_layer {args.skip_layer}, skip_sub_layer {args.skip_sub_layer}")
    else:
        print(f"prune method: {args.prune_method}")
    print(f"sparsity_ratio: {args.sparsity_ratio}")
    print(f"total_time: {total_time}\n\n")

    exit()

