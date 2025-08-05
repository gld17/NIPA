# CNN models
CUDA_VISIBLE_DEVICES=1 python main.py --model_name resnet18 --HWdes /share/guolidong-nfs/PIM/PA-SIM/HW_config/Device_Search_space.ini --dataset cifar10 --gpu_id 1 --sample_num 500 --log_file ./results/resnet18/cifar10_accsim.log --batch_size 1024 --cycle 10 & \
CUDA_VISIBLE_DEVICES=2 python main.py --model_name resnet18 --HWdes /share/guolidong-nfs/PIM/PA-SIM/HW_config/Device_Search_space.ini --dataset cifar100 --gpu_id 2 --sample_num 500 --log_file ./results/resnet18/cifar100_accsim.log --batch_size 1024 --cycle 10 & \
CUDA_VISIBLE_DEVICES=3 python main.py --model_name mobilenetv2 --HWdes /share/guolidong-nfs/PIM/PA-SIM/HW_config/Device_Search_space.ini --dataset cifar10 --gpu_id 3 --sample_num 500 --log_file ./results/mobilenetv2/cifar10_accsim.log --batch_size 1024 --cycle 10 & \
CUDA_VISIBLE_DEVICES=4 python main.py --model_name mobilenetv2 --HWdes /share/guolidong-nfs/PIM/PA-SIM/HW_config/Device_Search_space.ini --dataset cifar100 --gpu_id 4 --sample_num 500 --log_file ./results/mobilenetv2/cifar100_accsim.log --batch_size 1024 --cycle 10 & \
# CUDA_VISIBLE_DEVICES=5 python main.py --model_name efficientnetv2m --HWdes /share/guolidong-nfs/PIM/PA-SIM/HW_config/Device_Search_space.ini --dataset cifar10 --gpu_id 5 --sample_num 500 --log_file ./results/efficientnetv2m/cifar10_accsim.log --batch_size 1024 --cycle 10 & \
# CUDA_VISIBLE_DEVICES=6 python main.py --model_name efficientnetv2m --HWdes /share/guolidong-nfs/PIM/PA-SIM/HW_config/Device_Search_space.ini --dataset cifar100 --gpu_id 6 --sample_num 500 --log_file ./results/efficientnetv2m/cifar100_accsim.log --batch_size 1024 --cycle 10


# RWKV
# CUDA_VISIBLE_DEVICES=6 python main.py --model_name RWKV/rwkv-4-169m-pile --gpu_id 6 --sample_num 200 --log_file ./results/rwkv-4-169m-pile/piqa_accsim.log --dataset piqa --batch_size 128
# CUDA_VISIBLE_DEVICES=6 python main.py --model_name RWKV/rwkv-4-169m-pile --gpu_id 6 --sample_num 200 --log_file ./results/rwkv-4-169m-pile/arc_easy_accsim.log --dataset arc_easy --batch_size 64

# Mamba
# CUDA_VISIBLE_DEVICES=6 python main.py --model_name state-spaces/mamba-130m-hf --gpu_id 6 --sample_num 200 --log_file ./results/mamba-130m-hf/piqa_accsim.log --dataset piqa --batch_size 64
# CUDA_VISIBLE_DEVICES=6 python main.py --model_name state-spaces/mamba-130m-hf --gpu_id 6 --sample_num 200 --log_file ./results/mamba-130m-hf/arc_easy_accsim.log --dataset arc_easy --batch_size 64

# opt
# CUDA_VISIBLE_DEVICES=7 python main.py --model_name facebook/opt-125m --gpu_id 7 --sample_num 200 --log_file ./results/opt-125m/piqa_accsim.log --dataset piqa --batch_size 128
# CUDA_VISIBLE_DEVICES=7 python main.py --model_name facebook/opt-125m --gpu_id 7 --sample_num 200 --log_file ./results/opt-125m/arc_easy_accsim.log --dataset arc_easy --batch_size 128