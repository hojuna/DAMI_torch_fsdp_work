import os
import time
import datetime
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, DistributedSampler

from rich.console import Console
from rich.table import Table

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

    
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)  # 32x32x3 -> 30x30x32
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # 30x30x32 -> 28x28x64
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # 14x14x64 (after pooling)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))  # Conv1
        x = nn.functional.relu(self.conv2(x))  # Conv2
        x = nn.functional.max_pool2d(x, 2)     # Pooling (28x28x64 -> 14x14x64)
        x = torch.flatten(x, 1)                # Flatten to (batch_size, 64*14*14)
        x = nn.functional.relu(self.fc1(x))    # Fully connected layer 1
        x = self.fc2(x)                        # Fully connected layer 2
        return x

def log_memory_usage(rank):
    gpu_memory_allocated = torch.cuda.memory_allocated(rank) / (1024 ** 2)  # MB
    gpu_memory_reserved = torch.cuda.memory_reserved(rank) / (1024 ** 2)    # MB
    print(f"Rank {rank} - GPU Allocated: {gpu_memory_allocated:.2f} MB, Reserved: {gpu_memory_reserved:.2f} MB")

def train(rank, world_size):
    setup(rank, world_size)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 16

    # CIFAR-10 데이터 다운로드 및 로딩
    if rank == 0:
        torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    dist.barrier()  # 모든 프로세스가 다운로드 완료를 기다림

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, transform=transform)

    # 학습 및 검증 데이터셋 분할
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = random_split(trainset, [train_size, val_size])

    # 분산 샘플러 설정
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(testset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, sampler=test_sampler, num_workers=2)

    # 모델, 손실 함수, 옵티마이저 설정
    model = SimpleCNN().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01, momentum=0.9)

    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        ddp_model.train()
        train_sampler.set_epoch(epoch)

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(rank), target.to(rank)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Rank {rank}, Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        log_memory_usage(rank)
        epoch_end_time = time.time()
        print(f"Rank {rank}, Epoch {epoch+1}/{num_epochs} completed in {epoch_end_time - epoch_start_time:.2f} seconds")

    cleanup()
    
console = Console()

def log_epoch(rank, epoch, num_epochs, batch_idx, loss, epoch_time):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="center")
    table.add_column("Epoch", justify="center")
    table.add_column("Batch", justify="center")
    table.add_column("Loss", justify="right")
    table.add_column("Time (s)", justify="right")

    table.add_row(
        str(rank),
        f"{epoch}/{num_epochs}",
        str(batch_idx),
        f"{loss:.4f}",
        f"{epoch_time:.2f}"
    )
    console.print(table)

def log_training_start():
    console.rule("🚀 [bold green]Training Started[/bold green] 🚀", style="green")

def log_training_end(total_time):
    console.rule(f"✅ [bold blue]Training Completed in {total_time:.2f} seconds[/bold blue] ✅", style="blue")

def main():
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    total_start_time = time.time()
    print(f"Training started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    total_end_time = time.time()
    print(f"Training ended at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {total_end_time - total_start_time:.2f} seconds")

if __name__ == "__main__":
    main()