import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import time
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import numpy as np
import random


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    

def setup(rank, world_size):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 작업 그룹 초기화
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def load_data(rank, world_size, batch_size):
    """
    CIFAR-10 데이터 로드 및 분산 처리를 위한 DataLoader 설정
    """
    # 데이터 증강 및 정규화
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 데이터셋 로드
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    # 분산 샘플러 설정
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)
    
    # DataLoader 설정
    train_loader = DataLoader(
        dataset=train_dataset,
        num_workers=4,
        pin_memory=True,
        batch_sampler=batch_sampler_train,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, test_loader, train_sampler

def demo_basic(rank, world_size):

    batch_size = 128
    num_epochs = 100
    learning_rate = 0.1

    train_loader, test_loader, train_sampler = load_data(rank, world_size, batch_size)

    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    # 모델을 생성하고 순위 아이디가 있는 GPU로 전달
    model = SimpleCNN().to(rank)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=False)


    criterion  = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    dist.barrier()

    epoch_times = []

    for epoch in range(num_epochs):

        train_sampler.set_epoch(epoch)
        epoch_start_time = time.perf_counter()
        train_loss, train_acc = train_epoch(ddp_model, train_loader, criterion, optimizer, epoch, rank)
        test_loss, test_acc = validate(ddp_model, test_loader, criterion, rank)

        epoch_end_time = time.perf_counter()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        scheduler.step()
        if rank == 0:
            print(f'\nEpoch: {epoch}')
            print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}%')
            print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}%\n')
            print(f'Epoch Time: {epoch_time:.2f} seconds\n')

    #처음 값은 제외 ~ 뭔지 모르겠지만 처음에는 좀 오래 걸리는 것 같음
    epoch_times_tensor = torch.tensor(np.sum(epoch_times[1:])).to(rank)
    dist.all_reduce(epoch_times_tensor, op=dist.ReduceOp.SUM )

    if rank == 0:
        avg_epoch_time = epoch_times_tensor.item() / world_size / len(epoch_times[1:])
        print(f'\nAverage epoch time: {avg_epoch_time:.2f} seconds')

    cleanup()


def run_demo(demo_fn, world_size):

    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


def train_epoch(model, train_loader, criterion, optimizer, epoch, rank):
    """
    한 에폭 학습
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(rank, non_blocking=True), labels.to(rank, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if rank == 0 and batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                  f'Loss: {running_loss/(batch_idx+1):.3f}, '
                  f'Acc: {100.*correct/total:.3f}%')
    
    return running_loss / len(train_loader), 100. * correct / total

def validate(model, test_loader, criterion, rank):
    """
    검증 수행
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(rank, non_blocking=True), labels.to(rank, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return test_loss / len(test_loader), 100. * correct / total

if __name__ == "__main__":
    run_demo(demo_basic, 2)