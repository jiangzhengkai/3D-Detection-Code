#~/bin/bash

CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=2 --master_addr="127.0.0.1" --master_port=6666 ./tools/train.py --cfg configs/car_fhd.yaml
