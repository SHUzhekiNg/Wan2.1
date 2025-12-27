export CUDA_VISIBLE_DEVICES=1,2,3
conda activate vace && accelerate launch --config_file finetune/accelerate_config.yaml train_lora.py