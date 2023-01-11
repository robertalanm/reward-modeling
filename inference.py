import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


model = AutoModelForCausalLM.from_pretrained('robertmyers/bpt-sft')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
world_size = 4
ds_engine = deepspeed.init_inference(model,
                                        mp_size=world_size,
                                        dtype=torch.half,
                                        replace_method='auto',
                                        replace_with_kernel_inject=True)

model = ds_engine.module
# input_ids = tokenizer("Hello, my dog is cute", return_tensors="pt")

output = model('Human: what is the capitol of texas?')

# output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(output)
# check if on gpu 0 deepspeed, if so, import code; code.interact(local=locals())
    import code; code.interact(local=locals())