#~/bin/bash
python -m torch.distributed.launch --nproc_per_node=6 --master_addr="127.0.0.1" --master_port=8888 ./tools/train.py --cfg configs/car_fhd_kitti.yaml --model_dir kitti
