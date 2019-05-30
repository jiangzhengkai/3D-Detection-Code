#~/bin/bash
rlaunch --cpu 20 --gpu 4 --memory 150000 --  python -m torch.distributed.launch --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=8666 ./tools/train.py --cfg configs/all_nuscene_fhd.yaml --model_dir nusc_output
