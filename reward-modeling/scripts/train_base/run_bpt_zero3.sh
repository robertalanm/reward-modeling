deepspeed --num_gpus=8 finetune_base.py --config_path ../configs/base_configs/bpt.yaml \
--ds_config_path ../configs/ds_configs/ds_config_bpt_zero3.json \
--deepspeed ../configs/ds_configs/ds_config_bpt_zero3.json