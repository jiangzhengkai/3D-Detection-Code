#~/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=8669 ./tools/train.py --cfg configs/car_fhd_kitti.yaml --model_dir kitti_4card
