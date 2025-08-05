# CUDA_VISIBLE_DEVICES=2 python oracle.py --model_name vgg16 --dataset cifar10 --log_file ./results/oracle/vgg16_cifar10.log --gpu_id 2 & \
# CUDA_VISIBLE_DEVICES=2 python oracle.py --model_name vgg16 --dataset cifar100 --log_file ./results/oracle/vgg16_cifar100.log --gpu_id 2 & \
# CUDA_VISIBLE_DEVICES=3 python oracle.py --model_name resnet18 --dataset cifar10 --log_file ./results/oracle/resnet18_cifar10.log --gpu_id 3 & \
# CUDA_VISIBLE_DEVICES=3 python oracle.py --model_name resnet18 --dataset cifar100 --log_file ./results/oracle/resnet18_cifar100.log --gpu_id 3 & \
CUDA_VISIBLE_DEVICES=2 python oracle.py --model_name mobilenetv2 --dataset cifar10 --log_file ./results/oracle/mobilenetv2_cifar10.log --gpu_id 2 & \
CUDA_VISIBLE_DEVICES=2 python oracle.py --model_name mobilenetv2 --dataset cifar100 --log_file ./results/oracle/mobilenetv2_cifar100.log --gpu_id 2 & \
CUDA_VISIBLE_DEVICES=4 python oracle.py --model_name efficientnetv2m --dataset cifar10 --log_file ./results/oracle/efficientnetv2m_cifar10.log --gpu_id 4 & \
CUDA_VISIBLE_DEVICES=4 python oracle.py --model_name efficientnetv2m --dataset cifar100 --log_file ./results/oracle/efficientnetv2m_cifar100.log --gpu_id 4 & \

CUDA_VISIBLE_DEVICES=5 python oracle.py --model_name state-spaces/mamba-130m-hf --dataset piqa --log_file ./results/oracle/mamba-130m_piqa.log --gpu_id 5 --batch_size 32 & \
CUDA_VISIBLE_DEVICES=5 python oracle.py --model_name state-spaces/mamba-130m-hf --dataset arc_easy --log_file ./results/oracle/mamba-130m_arc_easy.log --gpu_id 5 --batch_size 32 & \
CUDA_VISIBLE_DEVICES=6 python oracle.py --model_name RWKV/rwkv-4-169m-pile --dataset piqa --log_file ./results/oracle/rwkv-169m_piqa.log --gpu_id 6 --batch_size 32 & \
CUDA_VISIBLE_DEVICES=6 python oracle.py --model_name RWKV/rwkv-4-169m-pile --dataset arc_easy --log_file ./results/oracle/rwkv-169m_arc_easy.log --gpu_id 6 --batch_size 32 

