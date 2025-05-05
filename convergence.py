from simplellm.llama import LLamaFirstStage, LLamaStage, LLamaLastStage # get our models
from simplellm.gpt import GPTFirstStage, GPTStage
from simplellm.tokenizers import SPTokenizer, GPTTokenizer # get our tokenizer
from simplellm.dataloaders import TinyStories, OpenWebText, RedPyjamav2 # get our dataset
from simplellm.utils import State
from simplellm.losses import causalLLMLoss, perplexityLoss # our loss
from copy import deepcopy
from sys import argv
import random
random.seed(42)
State.set_seed(42)
from torch.optim import AdamW
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch import save, cuda, zeros_like, cat, mean, std
import torch
import torch.distributed as dist
import traceback
import os
import json
from time import time
from math import sqrt
import math

            
dmodel = 1024
num_heads = 16
n_layers_per_stage = 4
n_stages = 6
seq_l = 1024
batch_size = 16
mb_count = 4
validation_amount = 20
max_iterations = 100001
init_lr = 3e-4
dp_size = 4

    
# make the tokenizer
def make_optim(params,lr):
    
    return AdamW(params, lr, betas=(0.9, 0.97), weight_decay=0.0)

tokenizer = SPTokenizer()
mesh = []
optimizers = []
dp_stage = []
optimizers_stage = []
for i in range(dp_size):
    torch.manual_seed(34107)
    torch.cuda.manual_seed(34107)
    # random.seed(0)
    s0 = LLamaFirstStage(tokenizer.vocab_size,dmodel=dmodel,num_heads=num_heads,
                        device="cuda:0", n_layers=0, ctx_size=seq_l,padding_idx=tokenizer.pad_id,de_embed=True)
    dp_stage.append(s0)
    if i > 0:
        dp_stage[-1].load_state_dict(deepcopy(dp_stage[0].state_dict()))
    optimizers_stage.append(make_optim(s0.parameters(),init_lr))
    
mesh.append(dp_stage)
optimizers.append(optimizers_stage)

    

for i in range(n_stages):
    dp_stage = []
    optimizers_stage = []
    for k in range(dp_size):
        torch.manual_seed(34107)
        torch.cuda.manual_seed(34107)
        # random.seed(0)
        dp_stage.append(LLamaStage(dmodel=dmodel,num_heads=num_heads,
                    device=f"cuda:{i+1}", n_layers=n_layers_per_stage, ctx_size=seq_l,padding_idx=tokenizer.pad_id))
        if k > 0:
            dp_stage[-1].load_state_dict(deepcopy(dp_stage[0].state_dict()))
        optimizers_stage.append(make_optim(dp_stage[-1].parameters(),init_lr))
        
    mesh.append(dp_stage)
    optimizers.append(optimizers_stage)

ds = OpenWebText(tokenizer,batch_size=batch_size, seq_l=seq_l,skip=validation_amount*2)
validation_dataset = OpenWebText(tokenizer,batch_size=16, seq_l=seq_l)


# we can iterate the dataset with:
iter_ds = iter(ds)



# used for dp communication
vls = []
once = True
for s in mesh:
    sizes = []
    len_sizes = []
    
    for param in s[0].parameters():
        sizes.append(param.shape)
        len_sizes.append(len(param.view(-1)))
    vls.append((sizes,len_sizes))
    
print(len(mesh))
print(len(mesh[0]))

n_stages += 1

for idx_stage in range(n_stages):
    mesh_weights = []
    for idx_dp in range(dp_size):
        tmp = []
        print("params of ",idx_stage,idx_dp)
        for param in mesh[idx_stage][idx_dp].parameters():
            tmp.append(param.to("cpu").view(-1))
        mesh_weights.append(torch.cat(tmp))
    mesh_weights_tmp = torch.cat(mesh_weights)
    print("PRE",idx_stage,"STD",torch.mean(torch.std(mesh_weights_tmp,dim=0)))
    mesh_weights_tmp = torch.mean(mesh_weights_tmp,dim=0)
    max_val = 0
    for idx in range(dp_size):
        max_val = max(max_val, torch.max((mesh_weights[idx] - mesh_weights_tmp).abs(),dim=0)[0])
    print("PRE",idx_stage,"MAX",max_val)

for itr in range(max_iterations):
    try:
        for s_optim in optimizers:
            for optim in s_optim:
                optim.zero_grad()
       
        
        for stage in range(dp_size):
            this_round_loss = 0
            for mbid in range(mb_count): 
                
                x = next(iter_ds)
                x = x.to("cuda:0")
                target = x.clone().detach()

                for i in range(n_stages):
                    
                    if i == 0:
                        x = mesh[i][stage].embed(x)
                    else:
                        x = x.to(f"cuda:{i}")
                        x = mesh[i][stage](x)
                x = x.to(f"cuda:0")
                x = mesh[0][stage].forward_end(x)
                loss = causalLLMLoss(x,target,tokenizer.vocab_size)
                loss = loss / mb_count
                this_round_loss += loss.item()
                loss.backward()
            print(stage,itr,this_round_loss)
        

        for s_optim in optimizers:
            for optim in s_optim:
                optim.step() 
        tmp = []
        if itr % 200 == 0:
            
            for idx_stage in range(n_stages):
                mesh_weights = []
                for idx_dp in range(dp_size):
                    tmp = []
                    for param in mesh[idx_stage][idx_dp].parameters():
                        tmp.append(param.data.to("cpu").view(-1))
                    mesh_weights.append(torch.cat(tmp))
                mesh_weights_tmp = torch.cat(mesh_weights)
                print(itr,idx_stage,"STD",torch.mean(torch.std(mesh_weights_tmp,dim=0)))
                mesh_weights_tmp = torch.mean(mesh_weights_tmp,dim=0)
                max_val = 0
                for idx in range(dp_size):
                    max_val = max(max_val, torch.max((mesh_weights[idx] - mesh_weights_tmp).abs(),dim=0)[0])
                print(itr,idx_stage,"MAX",max_val)

            

            
        cuda.empty_cache()
    except StopIteration:
        iter_ds = iter(ds)
    except Exception:
        print(traceback.format_exc())
        exit()




