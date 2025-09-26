CUDA_VISIBLE_DEVICES=2 python validate.py --model_name resnet18 --dataset cifar10 --gpu_id 2
CUDA_VISIBLE_DEVICES=2 python validate.py --model_name resnet18 --dataset cifar100 --gpu_id 2
CUDA_VISIBLE_DEVICES=2 python validate.py --model_name mobilenetv2 --dataset cifar10 --gpu_id 2
CUDA_VISIBLE_DEVICES=2 python validate.py --model_name mobilenetv2 --dataset cifar100 --gpu_id 2
CUDA_VISIBLE_DEVICES=2 python validate.py --model_name efficientnetv2m --dataset cifar10 --gpu_id 2
CUDA_VISIBLE_DEVICES=2 python validate.py --model_name efficientnetv2m --dataset cifar100 --gpu_id 2

CUDA_VISIBLE_DEVICES=2 python validate.py --model_name mamba-130m-hf --dataset piqa --gpu_id 2
CUDA_VISIBLE_DEVICES=2 python validate.py --model_name mamba-130m-hf --dataset arc_easy --gpu_id 2
CUDA_VISIBLE_DEVICES=2 python validate.py --model_name rwkv-4-169m-pile --dataset piqa --gpu_id 2
CUDA_VISIBLE_DEVICES=2 python validate.py --model_name rwkv-4-169m-pile --dataset arc_easy --gpu_id 2
CUDA_VISIBLE_DEVICES=2 python validate.py --model_name opt-125m --dataset piqa --gpu_id 2
CUDA_VISIBLE_DEVICES=2 python validate.py --model_name opt-125m --dataset arc_easy --gpu_id 2

CUDA_VISIBLE_DEVICES=2 python validate.py --model_name opt-125m --dataset piqa --cycle 3 --specific_layer layer1 --path /share/guolidong-nfs/PIM/PA-SIM/results/opt-125m/layer1
CUDA_VISIBLE_DEVICES=2 python validate.py --model_name opt-125m --dataset arc_easy --cycle 3 --specific_layer layer1 --path /share/guolidong-nfs/PIM/PA-SIM/results/opt-125m/layer1
CUDA_VISIBLE_DEVICES=2 python validate.py --model_name opt-125m --dataset piqa --cycle 3 --specific_layer layer2 --path /share/guolidong-nfs/PIM/PA-SIM/results/opt-125m/layer2
CUDA_VISIBLE_DEVICES=2 python validate.py --model_name opt-125m --dataset arc_easy --cycle 3 --specific_layer layer2 --path /share/guolidong-nfs/PIM/PA-SIM/results/opt-125m/layer2
CUDA_VISIBLE_DEVICES=0 python validate.py --model_name opt-1.3b --dataset piqa --cycle 3 --specific_layer layer21 --path /share/guolidong-nfs/PIM/PA-SIM/results/opt-1.3b/layer21
CUDA_VISIBLE_DEVICES=0 python validate.py --model_name opt-1.3b --dataset arc_easy --cycle 3 --specific_layer layer21 --path /share/guolidong-nfs/PIM/PA-SIM/results/opt-1.3b/layer21
CUDA_VISIBLE_DEVICES=0 python validate.py --model_name opt-1.3b --dataset piqa --cycle 3 --specific_layer layer22 --path /share/guolidong-nfs/PIM/PA-SIM/results/opt-1.3b/layer22
CUDA_VISIBLE_DEVICES=0 python validate.py --model_name opt-1.3b --dataset arc_easy --cycle 3 --specific_layer layer22 --path /share/guolidong-nfs/PIM/PA-SIM/results/opt-1.3b/layer22
CUDA_VISIBLE_DEVICES=0 python validate.py --model_name opt-1.3b --dataset piqa --cycle 3 --specific_layer layer23 --path /share/guolidong-nfs/PIM/PA-SIM/results/opt-1.3b/layer23
CUDA_VISIBLE_DEVICES=0 python validate.py --model_name opt-1.3b --dataset arc_easy --cycle 3 --specific_layer layer23 --path /share/guolidong-nfs/PIM/PA-SIM/results/opt-1.3b/layer23