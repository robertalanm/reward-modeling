import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


model = AutoModelForCausalLM.from_pretrained('robertmyers/bpt-sft')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
world_size = 4
model = deepspeed.init_inference(model,
                                        mp_size=world_size,
                                        dtype=torch.float,
                                        replace_method='auto',
                    replace_with_kernel_inject=True)


input_ids = tokenizer("Hello, my dog is cute", return_tensors="pt")

output_ids = model(input_ids)

output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output)
# check if on gpu 0 deepspeed, if so, import code; code.interact(local=locals())
# import code; code.interact(local=locals())