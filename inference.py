import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from bittensor import tokenizer as bt_tokenizer


# model = AutoModelForCausalLM.from_pretrained('robertmyers/bpt-sft')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
# world_size = 1
# ds_engine = deepspeed.init_inference(model,
#                                         mp_size=world_size,
#                                         dtype=torch.half,
#                                         replace_method='auto',
#                                         replace_with_kernel_inject=True)

# model = ds_engine.module
# tokenizer = bt_tokenizer()
# input_ids = tokenizer("Human: what is a money-line bet?\n", return_tensors="pt")['input_ids'].to('cuda')
# import code; code.interact(local=locals())


# # import code; code.interact(local=locals())

# output = model(input_ids)
# import code; code.interact(local=locals())

# output = tokenizer.decode(output[0], skip_special_tokens=False)
# if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
#     print(output)
# # # check if on gpu 0 deepspeed, if so, import code; code.interact(local=locals())
#     import code; code.interact(local=locals())


tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpwt-j-6B')

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

generator = pipeline('text-generation', 
                     model='robertmyers/bpt-sft',
                     device=local_rank,
                     tokenizer=tokenizer,)

generator.model = deepspeed.init_inference(generator.model,
                                           mp_size=world_size,
                                           dtype=torch.float,
                                           replace_method='auto',
					   replace_with_kernel_inject=True)


output = generator("Human: what is a money-line bet?\n", max_length=512, do_sample=True, top_k=512, top_p=0.95, num_return_sequences=1, early_stopping=True)
