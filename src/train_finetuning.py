"""
Pipeline com Fine-Tuning Progressivo para CheXpert
Descongelamento gradual de camadas da DenseNet121

Autor: Gabriel Ribeiro
"""

import os
import csv
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torchvision import models

warnings.filterwarnings("ignore")

# Configurações
TRAIN_DF_PATH = r"E:\GabrielRibeiro\chexpert_project\data\CheXpert-v1.0-small\train.csv"
TEST_DF_PATH  = r"E:\GabrielRibeiro\chexpert_project\src\CheXclusion\CXP\testSet_SubjID.csv"
PATH_IMAGE    = r"E:\GabrielRibeiro\chexpert_project\data"

DISEASES = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

class CheXpertDataset(data.Dataset):
    """Dataset CheXpert com política U-Ones."""
    
    def __init__(self, dataframe, path_image, transform=None, policy='u-ones'):
        self.dataframe = dataframe.reset_index(drop=True)
        self.path_image = path_image
        self.transform = transform
        self.labels_columns = DISEASES
        self.policy = policy
        
    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        caminho_imagem = item["Path"]
        img_path = os.path.join(self.path_image, caminho_imagem).replace("/", os.sep)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Imagem não encontrada: {img_path}")

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        label = torch.zeros(len(self.labels_columns), dtype=torch.float32)
        for i, disease in enumerate(self.labels_columns):
            value = item[disease]
            if np.isnan(value):
                label[i] = 0.0
            elif value == -1.0:
                label[i] = 1.0 if self.policy == 'u-ones' else 0.0
            else:
                label[i] = float(value)

        return img, label, str(caminho_imagem)

    def __len__(self):
        return len(self.dataframe)

class DenseNet121_Regularized(nn.Module):
    """DenseNet121 com Dropout e BatchNorm."""
    
    def __init__(self, num_classes=14, dropout_rate=0.5):
        super().__init__()
        self.densenet = models.densenet121(weights="DEFAULT")
        num_ftrs = self.densenet.classifier.in_features
        
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.densenet(x)

class EarlyStopping:
    """Early stopping baseado em validation loss."""
    
    def __init__(self, patience=7, delta=0.001, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"Early Stopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def compute_auroc_fast(model, data_loader, device, num_classes=14):
    """Calcula AUC-ROC médio e por classe."""
    
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for imgs, labels, _ in data_loader:
            imgs = imgs.to(device)
            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_labels.append(labels.numpy())
            all_probs.append(probs)
    
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    
    aurocs = []
    for i in range(num_classes):
        if len(np.unique(all_labels[:, i])) > 1:
            aurocs.append(roc_auc_score(all_labels[:, i], all_probs[:, i]))
        else:
            aurocs.append(0.5)
    
    return np.mean(aurocs), aurocs

def get_transforms():
    """Data augmentation."""
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        normalize
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform

def plot_learning_curves(log_path, output_path):
    """Gera gráficos de Loss e AUC-ROC."""
    
    if not os.path.exists(log_path):
        print(f"Arquivo '{log_path}' não encontrado.")
        return

    df = pd.read_csv(log_path)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(df['epoch'], df['train_loss'], label='Treino', color='blue', linewidth=2)
    axes[0].plot(df['epoch'], df['val_loss'], label='Validação', color='red', linewidth=2)
    axes[0].set_title('Evolução da Loss (BCE)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    df_with_auc = df[df['train_auroc'] > 0]
    if len(df_with_auc) > 0:
        axes[1].plot(df_with_auc['epoch'], df_with_auc['train_auroc'], 
                    label='Treino', color='blue', linewidth=2, marker='o')
        axes[1].plot(df_with_auc['epoch'], df_with_auc['val_auroc'], 
                    label='Validação', color='red', linewidth=2, marker='o')
        axes[1].set_title('Evolução do AUC-ROC', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('AUC-ROC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Gráficos salvos em '{output_path}'")
    plt.close()

def train_model(random_seed, results_dir, debug_mode=False):
    """
    Treina modelo com fine-tuning progressivo.
    
    Estratégia:
    - Época 0-10: Treina apenas classificador (backbone congelado)
    - Época 10-20: Descongela DenseBlock3 e DenseBlock4
    - Época 20+: Fine-tuning completo (todas as camadas)
    
    Args:
        random_seed: Seed para reprodutibilidade
        results_dir: Diretório de saída
        debug_mode: Se True, usa subset reduzido
    
    Returns:
        dict com métricas finais
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Hiperparâmetros
    BATCH_SIZE = 64
    WORKERS = 6
    N_LABELS = len(DISEASES)
    NUM_EPOCHS = 50
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    DROPOUT_RATE = 0.5
    
    if debug_mode:
        print("MODO DEBUG: 1000 imagens, 10 épocas")
        NUM_EPOCHS = 10
        BATCH_SIZE = 32

    # Reprodutibilidade
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device} | Seed: {random_seed}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Carrega dados
    print("\nCarregando datasets...")
    train_df_full = pd.read_csv(TRAIN_DF_PATH)
    test_indices = pd.read_csv(TEST_DF_PATH, header=None)[0].values
    test_df = train_df_full.iloc[test_indices].copy().reset_index(drop=True)

    for df in [train_df_full, test_df]:
        if 'path' in df.columns:
            df.rename(columns={'path': 'Path'}, inplace=True)

    if debug_mode:
        train_df_full = train_df_full.sample(n=1000, random_state=random_seed).reset_index(drop=True)
    else:
        train_df_full = train_df_full.sample(n=100000, random_state=random_seed).reset_index(drop=True)

    stratify_col = (train_df_full['Cardiomegaly'] == 1.0).astype(int)
    train_df, val_df = train_test_split(
        train_df_full, 
        test_size=0.15,
        random_state=random_seed,
        stratify=stratify_col
    )
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    print(f"Split: Treino {len(train_df):,} | Validação {len(val_df):,}")

    # Data loaders
    train_transform, val_transform = get_transforms()
    train_dataset = CheXpertDataset(train_df, PATH_IMAGE, train_transform, policy='u-ones')
    val_dataset = CheXpertDataset(val_df, PATH_IMAGE, val_transform, policy='u-ones')
    
    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=WORKERS, 
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = data.DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=WORKERS, 
        pin_memory=True,
        persistent_workers=True
    )

    # Modelo
    print("\nConstruindo modelo com fine-tuning progressivo...")
    model = DenseNet121_Regularized(num_classes=N_LABELS, dropout_rate=DROPOUT_RATE)
    model = model.to(device)

    # Fine-tuning progressivo: congela backbone inicialmente
    print("\nEstratégia de fine-tuning:")
    print("  Fase 1 (0-10 épocas): Apenas classificador")
    print("  Fase 2 (10-20 épocas): + DenseBlock3 e DenseBlock4")
    print("  Fase 3 (20+ épocas): Todas as camadas")
    
    for param in model.densenet.features.parameters():
        param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parâmetros treináveis iniciais: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    pos_weights = torch.ones(N_LABELS).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights).to(device)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, 
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-7
    )
    
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=7, delta=0.001)

    # Treinamento
    best_auroc = 0.0
    log_file_path = os.path.join(results_dir, "log_train.csv")
    
    with open(log_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_auroc", "val_auroc", "lr", "phase"])

    print("\nIniciando treinamento...")
    
    for epoch in range(NUM_EPOCHS):
        # Descongelamento progressivo
        current_phase = "Fase 1: Classificador"
        
        if epoch == 10:
            print("\n" + "="*70)
            print("FASE 2: Descongelando DenseBlock3 e DenseBlock4")
            print("="*70)
            
            for param in model.densenet.features.denseblock3.parameters():
                param.requires_grad = True
            for param in model.densenet.features.denseblock4.parameters():
                param.requires_grad = True
            
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=3e-5,
                weight_decay=WEIGHT_DECAY
            )
            
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Parâmetros treináveis: {trainable:,} ({100*trainable/total:.1f}%)")
            print(f"  Learning rate: {3e-5:.2e}")
            current_phase = "Fase 2: + DenseBlocks"
        
        if epoch == 20:
            print("\n" + "="*70)
            print("FASE 3: Fine-tuning completo")
            print("="*70)
            
            for param in model.parameters():
                param.requires_grad = True
            
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-5,
                weight_decay=WEIGHT_DECAY
            )
            
            print(f"  Todas as camadas desbloqueadas")
            print(f"  Learning rate: {1e-5:.2e}")
            current_phase = "Fase 3: Full Fine-Tuning"
        
        print(f"\nÉpoca {epoch+1}/{NUM_EPOCHS} - {current_phase}")
        
        # Treino
        model.train()
        running_loss_train = 0.0
        
        for imgs, labels, _ in tqdm(train_loader, desc="Treino"):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss_train += loss.item() * imgs.size(0)

        epoch_loss_train = running_loss_train / len(train_df)

        # Validação
        model.eval()
        running_loss_val = 0.0
        
        with torch.no_grad():
            for imgs, labels, _ in tqdm(val_loader, desc="Validação"):
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                with autocast(enabled=(device.type == 'cuda')):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                
                running_loss_val += loss.item() * imgs.size(0)

        epoch_loss_val = running_loss_val / len(val_df)

        # Métricas
        if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:
            train_auroc, _ = compute_auroc_fast(model, train_loader, device, N_LABELS)
            val_auroc, _ = compute_auroc_fast(model, val_loader, device, N_LABELS)
        else:
            train_auroc, val_auroc = 0.0, 0.0

        scheduler.step(epoch_loss_val)

        print(f"Loss - Treino: {epoch_loss_train:.4f} | Val: {epoch_loss_val:.4f} | Gap: {abs(epoch_loss_train - epoch_loss_val):.4f}")
        if train_auroc > 0:
            print(f"AUC-ROC - Treino: {train_auroc:.4f} | Val: {val_auroc:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        with open(log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, epoch_loss_train, epoch_loss_val, train_auroc, val_auroc, 
                           optimizer.param_groups[0]['lr'], current_phase])

        if val_auroc > best_auroc and val_auroc > 0:
            best_auroc = val_auroc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auroc': best_auroc,
                'best_val_loss': epoch_loss_val,
                'random_seed': random_seed
            }
            torch.save(checkpoint, os.path.join(results_dir, 'best_model.pt'))
            print(f"Melhor modelo salvo! AUC-ROC: {best_auroc:.4f}")

        early_stopping(epoch_loss_val)
        if early_stopping.early_stop:
            print("Early stopping ativado.")
            break

    print(f"\nTreinamento concluído. Melhor AUC-ROC: {best_auroc:.4f}")
    plot_learning_curves(log_file_path, os.path.join(results_dir, 'curvas_aprendizado.png'))
    
    return {
        'best_auroc': best_auroc,
        'random_seed': random_seed,
        'model_path': os.path.join(results_dir, 'best_model.pt'),
        'log_path': log_file_path
    }

if __name__ == "__main__":
    result = train_model(
        random_seed=85,
        results_dir = os.path.join('..', 'results_finetuning'),
        debug_mode=False
    )
    print(f"\nResultado final: AUC {result['best_auroc']:.4f}")
