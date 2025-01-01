import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import T5ForConditionalGeneration
import functools
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.t5.modeling_t5 import T5Block

# Hugging Face 캐시 경로 설정
HF_HOME = os.getenv("HF_HOME", "./hf_models")
os.environ["HF_HOME"] = HF_HOME
os.makedirs(HF_HOME, exist_ok=True)

# DummyDataset 정의
class DummyDataset(Dataset):
    def __init__(self, num_samples=64000, num_tokens=256, max_len=256, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.num_samples = num_samples
        self.input_ids = torch.randint(0, num_tokens, size=(num_samples, max_len))
        self.decoder_input_ids = torch.randint(0, num_tokens, size=(num_samples, max_len))
        self.labels = torch.randint(0, num_tokens, size=(num_samples, max_len))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "decoder_input_ids": self.decoder_input_ids[idx],
            "labels": self.labels[idx],
        }

# 최대 배치 크기 찾기
def find_max_batch_size(rank, world_size, start_batch_size=2, max_batch_size=1024):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:33445",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)

    model = T5ForConditionalGeneration.from_pretrained("t5-large", cache_dir=HF_HOME).to(rank)
    model = DDP(model, device_ids=[rank])

    dataset = DummyDataset(num_samples=1000)
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)

    batch_size = start_batch_size
    success_batch_size = 0

    # 지수적으로 배치 크기 증가
    while batch_size <= max_batch_size:
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        try:
            torch.cuda.empty_cache()
            for data in dataloader:
                data = {k: data[k].to(rank) for k in data}
                output = model(**data)
                torch.mean(output.loss).backward()
                break  # 한 번의 배치만 테스트
            print(f"Rank {rank}: Batch size {batch_size} succeeded.")
            success_batch_size = batch_size
            batch_size *= 2  # 배치 크기 두 배로 증가
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"Rank {rank}: Batch size {batch_size} failed due to OOM.")
                break
            else:
                raise e

    dist.destroy_process_group()
    return success_batch_size

# 학습 함수
def train(rank, world_size, batch_size, epochs=5):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:33445",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)
    model = T5ForConditionalGeneration.from_pretrained("t5-large", cache_dir=HF_HOME)
    wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={T5Block})
    model = FSDP(model, auto_wrap_policy=wrap_policy, device_id=rank)

    dataset = DummyDataset()
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in tqdm(range(epochs), desc=f"GPU {rank}: Epoch Progress", position=rank, leave=True):
        sampler.set_epoch(epoch)
        batch_bar = tqdm(dataloader, desc=f"GPU {rank}: Batch Progress", position=rank + world_size, leave=False)
        for data in batch_bar:
            data = {k: data[k].to(rank) for k in data}

            optimizer.zero_grad()
            output = model(**data)
            loss = output.loss
            loss.backward()
            optimizer.step()

            # 배치 진행 상황 표시
            batch_bar.set_postfix(loss=loss.item())

    dist.destroy_process_group()

# 메인 실행 함수
def main(rank, world_size, start_batch_size, max_batch_size, epochs):
    # 최대 배치 크기 탐색
    if rank == 0:
        print("Finding maximum batch size...")
    max_batch_size = find_max_batch_size(rank, world_size, start_batch_size, max_batch_size)

    # 학습 시작
    if rank == 0:
        print(f"Starting training with batch size {max_batch_size}...")
    train(rank, world_size, batch_size=max_batch_size, epochs=epochs)

if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    world_size = num_gpus
    start_batch_size = 2
    max_batch_size = 512
    epochs = 5

    mp.spawn(main, nprocs=num_gpus, args=(world_size, start_batch_size, max_batch_size, epochs))