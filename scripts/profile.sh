CUDA_VISIBLE_DEVICES=1 python profiler.py --model_name resnet18 --dataset cifar10 --gpu_id 1
CUDA_VISIBLE_DEVICES=1 python profiler.py --model_name resnet34 --dataset cifar10 --gpu_id 1
CUDA_VISIBLE_DEVICES=1 python profiler.py --model_name mobilenetv2 --dataset cifar10 --gpu_id 1
CUDA_VISIBLE_DEVICES=1 python profiler.py --model_name efficientnetv2m --dataset cifar10 --gpu_id 1
CUDA_VISIBLE_DEVICES=1 python profiler.py --model_name state-spaces/mamba-130m-hf --dataset piqa --gpu_id 1
CUDA_VISIBLE_DEVICES=2 python profiler.py --model_name RWKV/rwkv-4-169m-pile --dataset piqa --gpu_id 2