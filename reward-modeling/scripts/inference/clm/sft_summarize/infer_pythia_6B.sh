accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset Dahoas/openai_summarize_tldr_human_eval \
--log_file pythia_6B_sft_summarize_eval \
--model_name jon-tow/pythia-6.9b-summarize-sft \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split train \
--batch_size 2