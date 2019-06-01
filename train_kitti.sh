#~/bin/bash
python -m torch.distributed.launch --nproc_per_node=8 --master_addr="127.0.0.1" --master_port=8668 ./tools/train.py --cfg configs/car_fhd_kitti.yaml --model_dir kitti_output_8batch
