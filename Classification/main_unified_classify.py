import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    vgg16, VGG16_Weights,
    # googlenet, GoogLeNet_Weights, # Optional, if needed
    # alexnet, AlexNet_Weights # Optional, if needed
)
from tqdm import tqdm
import os
import argparse
from PIL import Image, ImageDraw
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
from collections import Counter

# --- Custom Models ---

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*56*56, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class DeeperCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            # 6-layer CNN
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)   # 7 -> 3
        )
        self.classifier = nn.Sequential(
            nn.Linear(512*3*3, 256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# --- Configuration & Experiments Definitions ---

# Repository root is resolved automatically from this script's location.
# Structure: <repo_root>/Code for C4RUDE/Classification/main_unified_experiments.py
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

EXPERIMENTS = {
    'needle_concentration': {
        'description': "Needle Classification - Different Concentrations (35kHz)",
        'root_train': os.path.join(REPO_ROOT, "Data", "01 needle", "classification", "02_diff_concentrations_", "frequency_35kHz", "train"),
        'root_val':   os.path.join(REPO_ROOT, "Data", "01 needle", "classification", "02_diff_concentrations_", "frequency_35kHz", "val"),
        'root_test':  os.path.join(REPO_ROOT, "Data", "01 needle", "classification", "02_diff_concentrations_", "frequency_35kHz", "test_validation_of_generatedtrained_data"),
        'wanted_classes': ["2", "3", "4", "5", "6", "7", "8"],
        'mask_type': 'circle',
        'mask_ratio': 0.3,
        'save_name': 'best_model_needle_concentration.pth'
    },
}

OUTPUT_DIR = os.path.join(REPO_ROOT, "results", "Classification")

# --- Model Selection Helper ---

def get_model(model_name, num_classes, pretrained=True):
    print(f"Initializing model: {model_name} (Classses: {num_classes}, Pretrained: {pretrained})")
    
    if model_name == 'resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'resnet34':
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'resnet50':
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'densenet121':
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
    elif model_name == 'mobilenet_v2':
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == 'efficientnet_b0':
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == 'vgg16':
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = vgg16(weights=weights)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        
    elif model_name == 'simplecnn':
        model = SimpleCNN(num_classes=num_classes)
        
    elif model_name == 'deepercnn':
        model = DeeperCNN(num_classes=num_classes)
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model

# --- Utility Functions ---

def get_all_files(root):
    all_files = []
    for rootdir, _, files in os.walk(root):
        for f in files:
            all_files.append(os.path.join(rootdir, f))
    return set(all_files)

def check_overlap(train_dir, val_dir, test_dir):
    print("Checking for duplicate filenames across splits...")
    train_files = get_all_files(train_dir)
    val_files = get_all_files(val_dir)
    test_files = get_all_files(test_dir)
    
    tv_overlap = len(train_files & val_files)
    tt_overlap = len(train_files & test_files)
    vt_overlap = len(val_files & test_files)
    
    print(f"Train-Val Overlap: {tv_overlap}")
    print(f"Train-Test Overlap: {tt_overlap}")
    print(f"Val-Test Overlap: {vt_overlap}")
    
    if tv_overlap > 0 or tt_overlap > 0 or vt_overlap > 0:
        print("WARNING: Data leakage detected (duplicate filenames).")

class RemapDataset(Dataset):
    def __init__(self, subset, class_map):
        self.subset = subset
        self.class_map = class_map
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return x, self.class_map[y]
    def __len__(self):
        return len(self.subset)

# --- Custom Transforms ---

class CenterCircleMask:
    def __init__(self, radius_ratio=0.35):
        if not (0 < radius_ratio < 1):
            raise ValueError("radius_ratio must be between 0 and 1")
        self.radius_ratio = radius_ratio

    def __call__(self, img):
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        w, h = img.size
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        r = int(min(w, h) * self.radius_ratio)
        center = (w // 2, h // 2)
        draw.ellipse((center[0]-r, center[1]-r, center[0]+r, center[1]+r), fill=255)
        
        img = img.copy()
        img.putalpha(mask)
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img)
        return img.convert("RGB")

class CenterSquareMask:
    def __init__(self, side_ratio=0.35):
        if not (0 < side_ratio < 1):
            raise ValueError("side_ratio must be between 0 and 1")
        self.side_ratio = side_ratio

    def __call__(self, img):
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        w, h = img.size
        side = int(min(w, h) * self.side_ratio)
        center = (w // 2, h // 2)
        left = center[0] - side // 2
        top = center[1] - side // 2
        right = center[0] + side // 2
        bottom = center[1] + side // 2
        
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([left, top, right, bottom], fill=255)
        
        img = img.copy()
        img.putalpha(mask)
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img)
        return img.convert("RGB")

def get_transforms(mask_type, mask_ratio, is_train=True):
    # Standard ImageNet normalization essential for pretrained ResNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    transform_list = []
    
    # 1. Masking
    if mask_type == 'circle':
        transform_list.append(CenterCircleMask(radius_ratio=mask_ratio))
    elif mask_type == 'square':
        transform_list.append(CenterSquareMask(side_ratio=mask_ratio))
        
    # 2. Common Ops
    transform_list.append(transforms.Grayscale(num_output_channels=3)) # Ensure 3 channels
    transform_list.append(transforms.Resize([224, 224]))
    
    # 3. Augmentation (Train only)
    if is_train:
        transform_list.append(transforms.RandomHorizontalFlip())
        # transform_list.append(transforms.RandomRotation(20)) # Optional
        
    # 4. Final Conversion
    transform_list.append(transforms.ToTensor())
    transform_list.append(normalize)
    
    return transforms.Compose(transform_list)

# --- Training & Evaluation ---

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_path):
    best_val_loss = float("inf")
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)

        # Validation
        val_acc, val_loss = evaluate_model(model, val_loader, criterion, device, name="Val", verbose=False)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save unwrapped model
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), save_path)
            print(f"--> Best model saved to {save_path}")
            
    return history

def evaluate_model(model, loader, criterion, device, name="Test", verbose=True):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = 100 * correct / total
    
    if verbose:
        print(f"[{name}] Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}%")
        
    return accuracy, avg_loss

def generate_plots(history, save_path):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy', color='green')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plot_file = os.path.join(os.path.dirname(save_path), 'training_history.png')
    plt.savefig(plot_file)
    print(f"History plot saved to {plot_file}")

def generate_confusion_matrix(model, loader, device, class_names, save_path):
    all_preds = []
    all_labels = []
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    cm_file = os.path.join(os.path.dirname(save_path), 'confusion_matrix.png')
    plt.savefig(cm_file)
    print(f"Confusion matrix image saved to {cm_file}")

# --- Main Logic ---

def run_experiment(exp_name, model_name='resnet18'):
    cfg = EXPERIMENTS[exp_name]
    print(f"\n{'='*60}")
    print(f"Running Experiment: {cfg['description']}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    # Update save name specific to model
    base_save_name = cfg['save_name'].replace('best_model', f'best_{model_name}') \
                                    .replace('best_resnet18', f'best_{model_name}') # Fallback
    
    # 1. Setup Directories
    exp_output_dir = os.path.join(OUTPUT_DIR, exp_name, model_name) # Subfolder for model
    os.makedirs(exp_output_dir, exist_ok=True)
    save_path = os.path.join(exp_output_dir, base_save_name)
    
    # 2. Check overlap
    check_overlap(cfg['root_train'], cfg['root_val'], cfg['root_test'])
    
    # 3. Validation Transform (Same for Val and Test)
    train_transform = get_transforms(cfg['mask_type'], cfg['mask_ratio'], is_train=True)
    eval_transform = get_transforms(cfg['mask_type'], cfg['mask_ratio'], is_train=False)
    
    # 4. Load & Filter Datasets
    print("Loading datasets...")
    trainset_full = datasets.ImageFolder(cfg['root_train'], transform=train_transform)
    valset_full = datasets.ImageFolder(cfg['root_val'], transform=eval_transform)
    testset_full = datasets.ImageFolder(cfg['root_test'], transform=eval_transform)
    
    # Helper to filter subsets
    def filter_dataset(full_ds, wanted_names):
        all_classes = full_ds.classes
        target_indices = [i for i, name in enumerate(all_classes) if name in wanted_names]
        if not target_indices:
            raise ValueError(f"No matching classes found! Available: {all_classes}, Wanted: {wanted_names}")
            
        class_map = {old: new for new, old in enumerate(target_indices)}
        indices = [i for i, t in enumerate(full_ds.targets) if t in target_indices]
        
        subset = Subset(full_ds, indices)
        remapped_ds = RemapDataset(subset, class_map)
        return remapped_ds, [all_classes[i] for i in target_indices]

    trainset, class_names = filter_dataset(trainset_full, cfg['wanted_classes'])
    valset, _ = filter_dataset(valset_full, cfg['wanted_classes'])
    testset, _ = filter_dataset(testset_full, cfg['wanted_classes'])
    
    print(f"Classes: {class_names}")
    print(f"Samples - Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")
    
    # 5. Dataloaders
    batch_size = 128
    workers = 0
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers)
    
    # 6. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_classes = len(class_names)
    
    # Initializing model using key provided or default
    model = get_model(model_name, num_classes, pretrained=True)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    # 7. Training
    num_epochs = 10
    history = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_path)
    generate_plots(history, save_path)
    
    # 8. Final Evaluation (Load Best)
    if os.path.exists(save_path):
        print(f"Loading best model from {save_path}...")
        state_dict = torch.load(save_path, map_location=device)
        
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
    
    evaluate_model(model, test_loader, criterion, device, name="Final Test")
    generate_confusion_matrix(model, test_loader, device, class_names, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Acoustic AI Classification Experiments")
    parser.add_argument("--experiment", type=str, default="all", 
                        choices=["all"] + list(EXPERIMENTS.keys()),
                        help="Which experiment to run")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=['resnet18', 'resnet34', 'resnet50', 'densenet121', 
                                 'mobilenet_v2', 'efficientnet_b0', 'vgg16', 
                                 'simplecnn', 'deepercnn'],
                        help="Which model architecture to use")
    
    args = parser.parse_args()
    
    if args.experiment == "all":
        for exp in EXPERIMENTS.keys():
            try:
                run_experiment(exp, args.model)
            except Exception as e:
                print(f"Error running experiment {exp}: {e}")
    else:
        run_experiment(args.experiment, args.model)
