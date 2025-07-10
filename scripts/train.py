import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from datetime import datetime
import time

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from models.vit_model import create_vit_model

PROCESSED_DATA_DIR = 'data/processed'
OUTPUT_DIR = 'output'
NUM_CLASSES = 2  # 例: normal, pneumonia
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10
IMG_SIZE = 224

def train():
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_output_dir = os.path.join(OUTPUT_DIR, timestamp)
    model_save_dir = os.path.join(run_output_dir, 'models')
    os.makedirs(model_save_dir, exist_ok=True)
    print(f'---{run_output_dir}にモデルを保存します---')
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Nomalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    full_dataset = datasets.ImageFolder(PROCESSED_DATA_DIR, transform=transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device('mps')
    
    model = create_vit_model(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        train_loss, train_corrects = 0.0, 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backword()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data())
        
        train_loss = train_loss / len(train_dataset)
        train_acc = train_corrects.double() / len(train_dataset)
        
        model.eval()
        val_loss, val_corrects = 0.0, 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).logits
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data())
        
        val_loss = val_loss / len(val_dataset)
        val_acc  = val_corrects.double() / len(val_dataset)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(model_save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'---{best_model_path}にモデルを保存しました---')
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.2f}s")
        
    print(f'---トレーニングが完了しました---')
    print(f'---{run_output_dir}にモデルを保存しました---')

if __name__ == '__main__':
    train()