import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration
from torch.nn.parallel import DataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import warnings
import time

warnings.filterwarnings("ignore")

# Hugging Face 모델 다운로드를 위한 HF_HOME 설정
HF_HOME = os.getenv("HF_HOME", "./hf_models")  # 기본값은 ./hf_models
os.environ["HF_HOME"] = HF_HOME  # 환경 변수 설정
os.makedirs(HF_HOME, exist_ok=True)  # 경로가 없으면 생성

# DummyDataset: 시드 고정
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

def parse_args():
    parser = argparse.ArgumentParser(description="T5 Training with Parallelization Options")
    parser.add_argument(
        "--parallel",
        type=str,
        choices=["DP", "DDP", "FSDP"],
        default="DP",
        help="Parallelization method: DP (DataParallel), DDP (DistributedDataParallel), FSDP (FullyShardedDataParallel)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--start_batch_size",
        type=int,
        default=2,
        help="Starting batch size for maximum batch size search",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=1024,
        help="Maximum batch size to search",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        help="Distributed backend",
    )
    return parser.parse_args()

def is_distributed(parallel):
    return parallel in ["DDP", "FSDP"]

# 메모리 확인 및 최대 배치 크기 탐색
def find_max_batch_size(device, parallel, start_batch_size=2, max_batch_size=1024, backend="nccl"):
    if is_distributed(parallel):
        rank = dist.get_rank()
        if rank == 0:
            print(f"Finding maximum batch size with {parallel}...")
    else:
        rank = 0
        print("Finding maximum batch size with DP...")

    model = T5ForConditionalGeneration.from_pretrained(
        "t5-large", cache_dir=HF_HOME  # HF_HOME을 캐시 경로로 사용
    )
    
    if parallel == "DDP":
        model = model.to(device)
        model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None)
    elif parallel == "FSDP":
        model = model.to(device)
        model = FSDP(model)
    else:
        model = model.to(device)
        model = DataParallel(model)

    dataset = DummyDataset(num_samples=1000)  # 샘플 데이터 수 줄이기

    batch_size = start_batch_size
    success_batch_size = batch_size

    while batch_size <= max_batch_size:
        current_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        try:
            # GPU 캐시 초기화
            torch.cuda.empty_cache()
            for data in current_dataloader:
                data = {k: v.to(device) for k, v in data.items()}
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    output = model(**data)
                    loss = torch.mean(output.loss)
                loss.backward()
                optimizer.step()
                break  # 한 번만 실행 후 종료

            if is_distributed(parallel):
                if rank == 0:
                    print(f"Batch size {batch_size} succeeded.")
            else:
                print(f"Batch size {batch_size} succeeded.")

            success_batch_size = batch_size  # 성공하면 기록
            batch_size *= 2  # 지수 증가

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                if is_distributed(parallel):
                    if rank == 0:
                        print(f"Batch size {batch_size} failed due to OOM.")
                else:
                    print(f"Batch size {batch_size} failed due to OOM.")
                batch_size //= 2  # 실패 시 배치 크기 감소
                break
            else:
                print(f"Unexpected error occurred: {str(e)}")
                raise e

    if is_distributed(parallel):
        dist.barrier()

    if is_distributed(parallel):
        if rank == 0:
            print(f"Maximum batch size: {success_batch_size}")
    else:
        print(f"Maximum batch size: {success_batch_size}")
    return success_batch_size

# 학습 함수
def train(device, batch_size, epochs=1, parallel="DP", backend="nccl"):
    if parallel == "DDP":
        model = T5ForConditionalGeneration.from_pretrained(
            "t5-large", cache_dir=HF_HOME  # HF_HOME을 캐시 경로로 사용
        ).to(device)
        model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None)
    elif parallel == "FSDP":
        model = T5ForConditionalGeneration.from_pretrained(
            "t5-large", cache_dir=HF_HOME
        ).to(device)
        model = FSDP(model)
    else:
        model = T5ForConditionalGeneration.from_pretrained(
            "t5-large", cache_dir=HF_HOME  # HF_HOME을 캐시 경로로 사용
        ).to(device)
        model = DataParallel(model)

    dataset = DummyDataset()
    if parallel == "DDP":
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, sampler=train_sampler)
    elif parallel == "FSDP":
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler()

    total_tokens = batch_size * len(dataset) * epochs
    start_time = time.time()

    # 에포크 진행 상황을 tqdm으로 표시
    for epoch in tqdm(range(epochs), desc="Epoch Progress", leave=True):
        if parallel == "DDP":
            train_sampler.set_epoch(epoch)
        batch_bar = tqdm(dataloader, desc=f"Batch Progress (Epoch {epoch + 1}/{epochs})", leave=False)
        
        # 누적 변수 초기화
        cumulative_tokens = 0
        cumulative_time = 0

        for data in batch_bar:
            batch_start_time = time.time()
            data = {k: v.to(device) for k, v in data.items()}

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(**data)
                loss = torch.mean(output.loss)
            
            if torch.isnan(loss):
                print(f"Warning: Loss is NaN at epoch {epoch + 1}, batch size {batch_size}. Skipping backward step.")
                scaler.unscale_(optimizer)
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_time = time.time() - batch_start_time
            tokens_processed = data["input_ids"].numel()
            cumulative_tokens += tokens_processed
            cumulative_time += batch_time
            avg_tokens_per_sec = cumulative_tokens / cumulative_time if cumulative_time > 0 else 0

            # 배치 진행 상황 바에 손실 및 평균 토큰 속도 표시
            batch_bar.set_postfix(loss=loss.item(), avg_tokens_per_sec=f"{avg_tokens_per_sec:.2f}")

    end_time = time.time()
    total_time = end_time - start_time
    total_tokens_processed = len(dataset) * batch_size * epochs
    overall_tps = total_tokens_processed / total_time if total_time > 0 else 0

    if is_distributed(parallel):
        rank = dist.get_rank()
        if rank == 0:
            print(f"Training completed in {total_time:.2f} seconds.")
            print(f"Overall tokens per second: {overall_tps:.2f}")
    else:
        print(f"Training completed in {total_time:.2f} seconds.")
        print(f"Overall tokens per second: {overall_tps:.2f}")

def setup_distributed(backend="nccl"):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    else:
        dist.init_process_group(backend=backend, init_method='env://', rank=0, world_size=1)

def cleanup_distributed():
    dist.destroy_process_group()

# GPU 설정 및 최대 배치 크기 탐색 실행
def main():
    args = parse_args()

    if is_distributed(args.parallel):
        setup_distributed(backend=args.backend)
        rank = dist.get_rank()
        local_rank = int(os.getenv('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if rank == 0:
            print(f"Using device: {device}")
            print(f"Parallelization method: {args.parallel}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        print(f"Parallelization method: {args.parallel}")

    max_batch_size = find_max_batch_size(
        device=device,
        parallel=args.parallel,
        start_batch_size=args.start_batch_size,
        max_batch_size=args.max_batch_size,
        backend=args.backend
    )

    if is_distributed(args.parallel):
        if dist.get_rank() == 0:
            print(f"Starting training with batch size {max_batch_size}...")
    else:
        print(f"Starting training with batch size {max_batch_size}...")

    train(
        device=device,
        batch_size=max_batch_size,
        epochs=args.epochs,
        parallel=args.parallel,
        backend=args.backend
    )

    if is_distributed(args.parallel):
        cleanup_distributed()

if __name__ == "__main__":
    main()