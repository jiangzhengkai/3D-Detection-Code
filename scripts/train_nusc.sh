#~/bin/bash
python -m torch.distributed.launch --nproc_per_node=8 --master_addr="127.0.0.1" --master_port=2222 ./tools/train.py --cfg configs/nuscenes/voxelnet/all_nuscene_fhd.yaml --model_dir /data/outputs/nusc_one_sweep
