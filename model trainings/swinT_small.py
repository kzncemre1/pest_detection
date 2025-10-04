import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os
import timm



def extract_fields(file_path):
    fields = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)
            if len(parts) > 1:
                fields.append(parts[1])
    return fields

file_path = "/deneme_projesi/classes.txt"
class_names = extract_fields(file_path)

config = {
    'train_data_dir': '/deneme_projesi/archive/goruntuler/train',
    'val_data_dir': '/deneme_projesi/archive/goruntuler/val',
    'test_data_dir': '/deneme_projesi/archive/goruntuler/test',
    'batch_size': 10,
    'epochs': 10,
    'lr': 3e-4,
    'num_workers': 2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_save_path': 'swinT_small_model_tam.pth',
    'class_names': class_names,
}


train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


from torch.utils.data import Subset
import random

def load_datasets():
    def get_partial_dataset(dataset, portion=1):
        total_size = len(dataset)
        subset_size = int(total_size * portion)
        indices = list(range(total_size))
        random.seed(42)  
        random.shuffle(indices)
        selected_indices = indices[:subset_size]
        return Subset(dataset, selected_indices)

    train_dataset = datasets.ImageFolder(config['train_data_dir'], train_transform)
    val_dataset = datasets.ImageFolder(config['val_data_dir'], val_transform)
    test_dataset = datasets.ImageFolder(config['test_data_dir'], val_transform)

    print(f"[INFO] Total train samples: {len(train_dataset)} | Using: {int( len(train_dataset))}")
    print(f"[INFO] Total val samples: {len(val_dataset)} | Using: {int(  len(val_dataset))}")
    print(f"[INFO] Total test samples: {len(test_dataset)} | Using: {int( len(test_dataset))}")

    train_loader = DataLoader(get_partial_dataset(train_dataset), 4, shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(get_partial_dataset(val_dataset), 4, shuffle=False, num_workers=config['num_workers'])
    test_loader = DataLoader(get_partial_dataset(test_dataset), 4, shuffle=False, num_workers=config['num_workers'])

    return train_loader, val_loader, test_loader

def initialize_model(num_classes):
    
    model = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
    
    
    if hasattr(model, 'head') and hasattr(model.head, 'fc') and isinstance(model.head.fc, nn.Linear):
        in_features = model.head.fc.in_features
        model.head.fc = nn.Linear(in_features, num_classes)

    
    elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    else:
        raise ValueError("Unexpected model structure: classifier/head/fc not found.")

    return model.to(config['device'])
    
    
def train_model(model, train_loader, val_loader):
    if model is None:
        model = initialize_model(len(config['class_names']))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    start_epoch = 0
    best_acc = 0.0

    if os.path.exists(config['model_save_path']):
        print(f"[INFO] Loading checkpoint from {config['model_save_path']}")
        checkpoint = torch.load(config['model_save_path'], map_location=config['device'])
        
        # Check if the checkpoint has the expected keys
        if 'state_dict' in checkpoint:  # Some models use 'state_dict' instead of 'model_state_dict'
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # If no key matches, try loading directly (some checkpoints are just state_dicts)
            try:
                model.load_state_dict(checkpoint)
            except:
                print("[WARNING] Couldn't load model from checkpoint. Starting fresh.")
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_acc' in checkpoint:
            best_acc = checkpoint['best_acc']

    




    total_train_batches = len(train_loader)
    total_val_batches = len(val_loader)

    for epoch in range(start_epoch, config['epochs']):
        print(f"\n[INFO] Epoch {epoch + 1}/{config['epochs']} started.")
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs.to(config['device']), labels.to(config['device'])

            optimizer.zero_grad()
            try:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

                if batch_idx % 10 == 0 or batch_idx == total_train_batches:
                    percent = 100 * batch_idx / total_train_batches
                    print(f"  [TRAIN] Batch {batch_idx}/{total_train_batches} ({percent:.1f}%)")

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("  [WARNING] Out of memory error caught. Skipping batch.")
                    torch.cuda.empty_cache()
                else:
                    raise e

        
        model.eval()
        val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader, 1):
                inputs, labels = inputs.to(config['device']), labels.to(config['device'])
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()

                if batch_idx % 10 == 0 or batch_idx == total_val_batches:
                    percent = 100 * batch_idx / total_val_batches
                    print(f"  [VAL] Batch {batch_idx}/{total_val_batches} ({percent:.1f}%)")

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, config['model_save_path'])
            print("  [INFO] New best model saved.")

        print(f"[RESULT] Epoch {epoch + 1}: Train Loss={epoch_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

def evaluate_model(model,test_loader):
    
    checkpoint = torch.load(config['model_save_path'], map_location=config['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config['device'])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(config['device'])
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=config['class_names']))

    import pandas as pd
    
    report = classification_report(all_labels, all_preds, target_names=config['class_names'], output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
   
    df_report.to_csv("swinT.csv", index=True)

def visualize_predictions(model,test_loader):
    

    checkpoint = torch.load(config['model_save_path'], map_location=config['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config['device'])
    model.eval()

    images, labels = next(iter(test_loader))
    images, labels = images[:10], labels[:10]
    images_cpu = images.cpu()

    with torch.no_grad():
        outputs = model(images.to(config['device']))
        _, preds = torch.max(outputs, 1)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flatten()):
        img = images_cpu[i].permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        ax.imshow(img)
        ax.set_title(f"Pred: {config['class_names'][preds[i]]}\nTrue: {config['class_names'][labels[i]]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    print(torch.cuda.is_available())
    train_loader, val_loader, test_loader = load_datasets()
    
    model = initialize_model(len(config['class_names']))

    #train_model(model, train_loader, val_loader)
    evaluate_model(model, test_loader)
    #visualize_predictions(model, test_loader)
