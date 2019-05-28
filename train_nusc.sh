#~/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=6 --master_addr="127.0.0.1" --master_port=8666 ./tools/train.py --cfg configs/all_nuscene_fhd.yaml --model_dir nusc_output
