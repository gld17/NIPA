# CUDA_VISIBLE_DEVICES=1 python wrap_error.py --model_name resnet18 --dataset cifar10 --gpu_id 1 --log_file ./results/resnet18/abs_cifar10_accsim.log --batch_size 256 & \
# CUDA_VISIBLE_DEVICES=2 python wrap_error.py --model_name resnet18 --dataset cifar100 --gpu_id 2 --log_file ./results/resnet18/abs_cifar100_accsim.log --batch_size 256 & \
# CUDA_VISIBLE_DEVICES=3 python wrap_error.py --model_name mobilenetv2 --dataset cifar10 --gpu_id 3 --log_file ./results/mobilenetv2/abs_cifar10_accsim.log --batch_size 256 & \
# CUDA_VISIBLE_DEVICES=4 python wrap_error.py --model_name mobilenetv2 --dataset cifar100 --gpu_id 4 --log_file ./results/mobilenetv2/abs_cifar100_accsim.log --batch_size 256 & \
# CUDA_VISIBLE_DEVICES=5 python wrap_error.py --model_name efficientnetv2m --dataset cifar10 --gpu_id 5 --log_file ./results/efficientnetv2m/abs_cifar10_accsim.log --batch_size 256 & \
# CUDA_VISIBLE_DEVICES=6 python wrap_error.py --model_name efficientnetv2m --dataset cifar100 --gpu_id 6 --log_file ./results/efficientnetv2m/abs_cifar100_accsim.log --batch_size 256

# CUDA_VISIBLE_DEVICES=1 python wrap_error.py --model_name state-spaces/mamba-130m-hf --dataset piqa --gpu_id 1 --log_file ./results/mamba-130m-hf/abs_piqa_accsim.log --batch_size 64 & \
# CUDA_VISIBLE_DEVICES=2 python wrap_error.py --model_name state-spaces/mamba-130m-hf --dataset arc_easy --gpu_id 2 --log_file ./results/mamba-130m-hf/abs_arc_easy_accsim.log --batch_size 64 & \
# CUDA_VISIBLE_DEVICES=3 python wrap_error.py --model_name RWKV/rwkv-4-169m-pile --dataset piqa --gpu_id 1 --log_file ./results/rwkv-4-169m-pile/abs_piqa_accsim.log --batch_size 64 & \
# CUDA_VISIBLE_DEVICES=4 python wrap_error.py --model_name RWKV/rwkv-4-169m-pile --dataset arc_easy --gpu_id 2 --log_file ./results/rwkv-4-169m-pile/abs_arc_easy_accsim.log --batch_size 64 & \
CUDA_VISIBLE_DEVICES=3 python wrap_error.py --model_name facebook/opt-125m --dataset piqa --gpu_id 3 --log_file ./results/opt-125m/abs_piqa_accsim.log --batch_size 256 & \
CUDA_VISIBLE_DEVICES=4 python wrap_error.py --model_name facebook/opt-125m --dataset arc_easy --gpu_id 4 --log_file ./results/opt-125m/abs_arc_easy_accsim.log --batch_size 256