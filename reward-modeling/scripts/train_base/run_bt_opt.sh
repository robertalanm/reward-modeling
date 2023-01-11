deepspeed --num_gpus=8 finetune_base.py --config_path ../configs/base_configs/bt_opt.yaml \
--ds_config_path ../configs/ds_configs/ds_config_bt_opt.json \
--deepspeed ../configs/ds_configs/ds_config_bt_opt.json