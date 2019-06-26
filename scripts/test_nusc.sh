python -m torch.distributed.launch --nproc_per_node=8 --master_addr="127.0.0.1" --master_port=8667 ./tools/test.py --cfg configs/nuscenes/voxelnet/all_nuscene_fhd.yaml --model_dir /data/outputs/nusc
