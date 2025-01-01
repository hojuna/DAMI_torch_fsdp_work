python t5_model.py --parallel DP
python /root/DAMI-Torch/sessions/session05_assignment/t5_model.py --parallel DP

python -m torch.distributed.launch --nproc_per_node=2 t5_model.py --parallel DDP
python t5_model.py --parallel FSDP

python t5_model.py --parallel DP --epochs 1 --max_batch_size 2048