import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, Subset
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy
import json
import argparse
from utils import load_yaml, load_jsonl, freeze_bottom_causal_layers
from rm_datasets import SFTDataset
import wandb
from datasets import load_dataset 
from torch.utils.data import DataLoader

from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed


def train(config):
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
    except ValueError:
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], use_fast=False)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    training_args = TrainingArguments(**config["train_args"])
    model = AutoModelForCausalLM.from_pretrained(config["model_path"]).cuda()



    data = load_dataset(config["data_path"], revision='v1.2-jazzy')["train"]


    data = data.filter(lambda e, i: i<1000, with_indices=True)
    print("Len data: ", len(data))


    dataset = SFTDataset(data, tokenizer)
    
    # reduce the examples to 1000 for debugging

    train_size = int(0.94 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    # split the train dataset down to 1000 examples for debugging
    # train_dataset, val_dataset = random_split(train_dataset, [1000, len(train_dataset) - 1000])

    Trainer(model=model, args=training_args, train_dataset=train_dataset,
            eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                                'attention_mask': torch.stack([f[1] for f in data]),
                                                                'labels': torch.stack([f[2] for f in data])}).train()


    # doing my own zero implementation
    # ds_config = deepspeed.parse_config(config["train_args"]["deepspeed"])
    # dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive
    # model = AutoModelForCausalLM.from_pretrained(config["model_path"]).cuda()
    # model_engine, optimizer, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=model.parameters(), config_params=ds_config)


    # # data loader

    # train_dataloader = DataLoader(
    #     train_dataset,
    #     collate_fn=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
    #                                 'attention_mask': torch.stack([f[1] for f in data]),
    #                                 'labels': torch.stack([f[2] for f in data])},
    #     batch_size=training_args.per_device_train_batch_size,

    # )

    # # train loop
    # for epoch in range(training_args.num_train_epochs):
    #     for step, batch in enumerate(train_dataloader):
    #         model_engine.train()
    #         model_engine.zero_grad()
    #         outputs = model_engine(**batch)
    #         loss = outputs[0]
    #         model_engine.backward(loss)
    #         model_engine.step()
    #         model_engine.zero_grad()
    #         if step % 10 == 0:
    #             print(f"Epoch: {epoch} Step: {step} Loss: {loss}")


    if torch.distributed.get_rank() == 0:
        if os.environ.get('DEEPSPEED_ZERO_STAGE', '0') != '3':
            EOS_ID = tokenizer("<|endoftext|>")["input_ids"][0]
            data = []
            for i in range(16):
                prompt = val_dataset[i][3]
                inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].view(1, -1).cuda()
                attention_mask = inputs["attention_mask"].view(1, -1).cuda()
                sample_outputs = model.generate(input_ids, attention_mask=attention_mask, do_sample=True, max_length=1024)
                response = tokenizer.batch_decode(sample_outputs)[0].split("<|endoftext|>")[0][len(prompt):]
                data.append([prompt, response])
            cols = ["prompt", "response"]
            wandb.log({"samples": wandb.Table(columns=cols, data=data)})

    model.save_pretrained('bpt-sft-base')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--ds_config_path", type=str)
    parser.add_argument("--deepspeed", type=str)
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    config = load_yaml(args.config_path)
    config["train_args"]["deepspeed"] = args.ds_config_path

    train(config)