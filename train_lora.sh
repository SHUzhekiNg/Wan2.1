export CUDA_VISIBLE_DEVICES=0
accelerate launch --config_file finetune/accelerate_config.yaml train_lora.py