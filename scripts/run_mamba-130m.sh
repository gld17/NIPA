# Mamba
CUDA_VISIBLE_DEVICES=7 python main.py --model_name state-spaces/mamba-130m-hf --gpu_id 7 --sample_num 300 --log_file ./results/mamba-130m-hf/piqa_accsim.log --dataset piqa --batch_size 128
CUDA_VISIBLE_DEVICES=7 python main.py --model_name state-spaces/mamba-130m-hf --gpu_id 7 --sample_num 300 --log_file ./results/mamba-130m-hf/arc_easy_accsim.log --dataset arc_easy --batch_size 128