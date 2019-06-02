#~/bin/bash
python -m torch.distributed.launch --nproc_per_node=8 --master_addr="127.0.0.1" --master_port=8667 ./tools/train.py --cfg configs/all_nuscene_fhd.yaml --model_dir nusc_tmp
