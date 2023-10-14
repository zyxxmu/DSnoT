import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import check_sparsity, prune_DSnoT, prune_magnitude, prune_sparsegpt, prune_wanda
from lib.prune_opt import check_sparsity_opt, prune_DSnoT_opt
from lib.eval import eval_ppl
from lib.save_results import save_ppl_result

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = 2048
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--model_type', type=str, default=None, help='model type, either llama or opt')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--eval_dataset', type=str, default='wikitext2', choices=["wikitext2", "c4", "ptb"], help='eval ppl on dataset')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity ratio.')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["wanda", "sparsegpt", "magnitude", "DSnoT", "dense"])
    parser.add_argument("--initial_method", type=str, choices=["wanda", "sparsegpt", "magnitude"])

    parser.add_argument('--max_cycle_time', type=int, default=10, help='Max cycle time.')
    parser.add_argument('--without_DSnoT', action="store_true", help="without DSnoT")
    parser.add_argument('--update_threshold', type=float, default=0, help='update threshold.')
    parser.add_argument('--pow_of_var_regrowing', type=float, default=0, help='The power of variance.')
    parser.add_argument('--pow_of_var_pruning', type=float, default=0, help='The power of variance.')
    parser.add_argument("--skip_layer", type=str, default="no_skip", choices=["no_skip", "mlp", "self_attn"])
    parser.add_argument("--skip_sub_layer", type=str, default="no_skip", choices=["no_skip", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "fc1", "fc2", "out_proj"])
    parser.add_argument('--without_same_sign', action="store_true", help="without same sign")
    
    parser.add_argument('--get_time_overhead', action="store_true", help="get time overhead")
    
    parser.add_argument("--output_results_file", default="results.txt", type=str)
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling different model types
    if not args.model_type:
        if ["llama kind" for model_name in ["llama", "vicuna"] if model_name in args.model]:
            args.model_type = "llama"
        elif "opt" in args.model:
            args.model_type = "opt"
        else:
            Warning("Model type not specified from model path, please specify manually")
            exit()
    print(f"model type: {args.model_type}")

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)
    
    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.model_type == "llama":
            if args.prune_method == "wanda":
                prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "magnitude":
                prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "sparsegpt":
                prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "DSnoT":
                prune_DSnoT(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "dense":
                pass
        elif args.model_type == "opt":
            if args.prune_method == "wanda":
                prune_wanda_opt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "magnitude":
                prune_magnitude_opt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "sparsegpt":
                prune_sparsegpt_opt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "DSnoT":
                prune_DSnoT_opt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "dense":
                pass
    
    ################################################################
    print("*"*30)
    if args.model_type == "llama":
        sparsity_ratio = check_sparsity(model)
    elif args.model_type == "opt":
        sparsity_ratio = check_sparsity_opt(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    
    dataset = 'wikitext2'
    ppl = eval_ppl(model, tokenizer, dataset, device)
    print(f"\nppl on {dataset}: {ppl}\n")

    save_ppl_result(args, args.output_results_file, sparsity_ratio, ppl)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()