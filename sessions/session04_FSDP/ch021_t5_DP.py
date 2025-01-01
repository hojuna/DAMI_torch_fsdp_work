import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration
from torch.nn.parallel import DataParallel

import warnings
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

# 메모리 확인 및 최대 배치 크기 탐색
def find_max_batch_size(device, start_batch_size=2, max_batch_size=1024):
    model = T5ForConditionalGeneration.from_pretrained(
        "t5-large", cache_dir=HF_HOME  # HF_HOME을 캐시 경로로 사용
    )
    model.to(device)
    model = DataParallel(model)  # DP 설정
    dataset = DummyDataset(num_samples=1000)  # 샘플 데이터 수 줄이기
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    batch_size = start_batch_size
    success_batch_size = batch_size

    while batch_size <= max_batch_size:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        try:
            # GPU 캐시 초기화
            torch.cuda.empty_cache()
            for data in dataloader:
                data = {k: data[k].to(device) for k in data}
                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda'):
                    output = model(**data)
                    loss = torch.mean(output.loss)
                loss.backward()
                optimizer.step()
                break  # 한 번만 실행 후 종료

            print(f"Batch size {batch_size} succeeded.")
            success_batch_size = batch_size  # 성공하면 기록
            batch_size *= 2  # 지수 증가

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"Batch size {batch_size} failed due to OOM.")
                batch_size //= 2  # 실패 시 배치 크기 감소
                break
            else:
                print(f"Unexpected error occurred: {str(e)}")
                raise e

    print(f"Maximum batch size: {success_batch_size}")
    return success_batch_size

# 학습 함수
def train(device, batch_size, epochs=1):
    model = T5ForConditionalGeneration.from_pretrained(
        "t5-large", cache_dir=HF_HOME  # HF_HOME을 캐시 경로로 사용
    )
    model.to(device)
    model = DataParallel(model)  # DP 설정

    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scaler = torch.amp.GradScaler()

    # 에포크 진행 상황을 tqdm으로 표시
    for epoch in tqdm(range(epochs), desc="Epoch Progress", leave=True):
        batch_bar = tqdm(dataloader, desc=f"Batch Progress (Epoch {epoch + 1}/{epochs})", leave=False)
        for data in batch_bar:
            data = {k: data[k].to(device) for k in data}

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                output = model(**data)
                loss = torch.mean(output.loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 배치 진행 상황 바에 손실 표시
            batch_bar.set_postfix(loss=loss.item())

# GPU 설정 및 최대 배치 크기 탐색 실행
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Finding maximum batch size...")
    max_batch_size = find_max_batch_size(device=device, start_batch_size=2, max_batch_size=1024)

    print(f"Starting training with batch size {max_batch_size}...")
    train(device=device, batch_size=max_batch_size, epochs=5)