accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset Dahoas/synthetic_prompt_responses \
--log_file 6B_rm_on_synthetic_test \
--model_name EleutherAI/gpt-j-6B \
--tokenizer_name EleutherAI/gpt-j-6B \
--split test \
--batch_size 4 \
--rm_path /fsx/alex/ckpts/gptj-rm/good_ckpt/hf_ckpt.pt \
--order "instruct" "20B" "6B" "1B" "125M"