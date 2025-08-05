# RWKV
CUDA_VISIBLE_DEVICES=7 python main.py --model_name RWKV/rwkv-4-169m-pile --gpu_id 7 --sample_num 300 --log_file ./results/rwkv-4-169m-pile/piqa_accsim.log --dataset piqa --batch_size 128
CUDA_VISIBLE_DEVICES=7 python main.py --model_name RWKV/rwkv-4-169m-pile --gpu_id 7 --sample_num 300 --log_file ./results/rwkv-4-169m-pile/arc_easy_accsim.log --dataset arc_easy --batch_size 256