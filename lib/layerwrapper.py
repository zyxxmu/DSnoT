import math

import torch
import torch.nn as nn
import transformers


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """
    def __init__(self, layer, initial_method = None, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        self.nsamples = 0

        self.initial_method = initial_method
        if self.initial_method == "sparsegpt":
            self.H = torch.zeros((self.columns, self.columns), device=self.dev)

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.sum_metric_row = torch.zeros((self.columns), device=self.dev)
        
        self.mean = torch.zeros((self.columns), device=self.dev)
        self.var = torch.zeros((self.columns), device=self.dev)
        self.ntokens = 0
        
        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        inp = inp.type(torch.float32)

        mean_inp = torch.mean(inp, dim=1, keepdim=True)

        var_inp = torch.var(inp, dim=1, unbiased=False, keepdim=True)
        num_inp = inp.shape[1]
        self.var = var_inp if self.ntokens == 0 else (self.var * self.ntokens + var_inp * num_inp) / (self.ntokens + num_inp)
        self.mean = mean_inp if self.ntokens == 0 else (self.mean * self.ntokens + mean_inp * num_inp) / (self.ntokens + num_inp)
        self.ntokens += num_inp
        
        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.sum_metric_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        self.sum_metric_row += torch.sum(inp, dim=1) / self.nsamples

    def free(self):
        self.H = None
        torch.cuda.empty_cache()
