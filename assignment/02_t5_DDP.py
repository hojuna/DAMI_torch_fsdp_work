import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import T5ForConditionalGeneration
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.cuda.amp import GradScaler

import warnings
warnings.filterwarnings("ignore")

HF_HOME = os.getenv("HF_HOME", "./hf_models")
os.environ["HF_HOME"] = HF_HOME
os.makedirs(HF_HOME, exist_ok=True)


class DummyDataset(Dataset):
    def __init__(self, num_samples=640, num_tokens=256, max_len=256, seed=42):
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


def find_max_batch_size(local_rank, model, dataset, device, start_batch_size=2, max_batch_size=1024):
    batch_size = start_batch_size
    success_batch_size = batch_size
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scaler = GradScaler()

    pbar = tqdm(total=max_batch_size, desc=f"GPU {local_rank}: Finding Max Batch Size", position=local_rank)

    while batch_size <= max_batch_size:
        sampler = DistributedSampler(dataset, shuffle=False)
        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=True)
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

        try:
            torch.cuda.empty_cache()
            for i in range(2):
                for data in dataloader:
                    data = {k: v.to(device) for k, v in data.items()}
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        output = model(**data)
                        loss = torch.mean(output.loss)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    break  # 배치 크기 테스트가 끝나면 다음으로 이동

            success_batch_size = batch_size
            batch_size += 1  # 1씩 증가
            pbar.set_postfix(success_batch_size=success_batch_size)

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"GPU {local_rank}: Batch size {batch_size} failed due to OOM.")
                batch_size -= 1  # OOM 발생 시 1씩 감소
                break
            else:
                raise e
            
    del dataloader  # 기존 dataloader 삭제
    del sampler    # 기존 sampler 삭제
    del optimizer  # 기존 optimizer 삭제
    del scaler     # 기존 scaler 삭제
    torch.cuda.empty_cache()  # 메모리 정리
    pbar.close()
    return success_batch_size


def train_ddp(local_rank, world_size, epochs=2):
    # DDP 초기화
    dist.init_process_group(backend="nccl", init_method="env://", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 데이터 및 모델 준비
    dataset = DummyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)



    model = T5ForConditionalGeneration.from_pretrained("t5-large", cache_dir=HF_HOME).to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scaler = GradScaler()

    # 배치 크기 탐색
    max_batch_size = find_max_batch_size(local_rank, model, dataset, device)
    print(f"GPU {local_rank}: Using batch size {max_batch_size}")
    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=max_batch_size, drop_last=True)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

    # 학습 루프
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # 샘플러 재설정

        batch_bar = tqdm(dataloader, desc=f"GPU {local_rank} Epoch {epoch + 1}/{epochs}", position=local_rank)
        for i, data in enumerate(batch_bar):
            data = {k: v.to(device) for k, v in data.items()}

            optimizer.zero_grad()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()  # 타이밍 시작

            with torch.cuda.amp.autocast():
                output = model(**data)
                loss = torch.mean(output.loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            end_time.record()  # 타이밍 종료
            torch.cuda.synchronize()  # GPU 연산 동기화

            # 처리된 토큰 계산
            batch_tokens = max_batch_size * data["input_ids"].size(1)
            elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # 초 단위로 변환
            tokens_per_second = batch_tokens / elapsed_time if elapsed_time > 0 else 0

            # tqdm 업데이트
            batch_bar.set_postfix(
                loss=loss.item(),
                tokens_per_second=f"{tokens_per_second:.2f}",
            )
            
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train_ddp, args=(world_size,), nprocs=world_size)