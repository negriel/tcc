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
import torchvision.transforms as transforms
from torchvision import models

warnings.filterwarnings("ignore")

TRAIN_DF_PATH = r"E:\GabrielRibeiro\chexpert_project\data\CheXpert-v1.0-small\train.csv"
TEST_DF_PATH  = os.path.join('..', 'results', 'chexclusion_test_split.csv')
PATH_IMAGE    = r"E:\GabrielRibeiro\chexpert_project\data"
RESULTS_DIR   = os.path.join('..', 'results_ensemble')

DISEASES = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]


class CheXpertDataset(data.Dataset):
    """Dataset CheXpert com política U-Ones para incertezas."""
    
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
    """DenseNet121 com Dropout e BatchNorm para reduzir overfitting."""
    
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


def get_transforms():
    """Transformações para validação."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    return val_transform

def load_model(model_path, device, num_classes=14, dropout_rate=0.5):
    """Carrega um modelo treinado."""
    model = DenseNet121_Regularized(num_classes=num_classes, dropout_rate=dropout_rate)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model

def compute_predictions(model, data_loader, device):
    """Computa predições de um modelo."""
    all_labels = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for imgs, labels, _ in tqdm(data_loader, desc="Computando predições"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_labels.append(labels.numpy())
            all_probs.append(probs)
    
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    
    return all_labels, all_probs

def compute_auroc(labels, probs, num_classes=14):
    """Calcula AUC-ROC por classe."""
    aurocs = []
    for i in range(num_classes):
        if len(np.unique(labels[:, i])) > 1:
            aurocs.append(roc_auc_score(labels[:, i], probs[:, i]))
        else:
            aurocs.append(0.5)
    
    return aurocs

def plot_results(mean_auroc, aurocs, diseases, output_dir):
    """Gera gráfico de barras com AUC por doença."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = []
    for auc in aurocs:
        if auc >= 0.80:
            colors.append('#2ecc71')  
        elif auc >= 0.70:
            colors.append('#3498db')
        else:
            colors.append('#e74c3c') 
    
    bars = ax.barh(diseases, aurocs, color=colors, edgecolor='black', linewidth=1.2)
    
    ax.axvline(mean_auroc, color='red', linestyle='--', linewidth=2, 
               label=f'Média: {mean_auroc:.4f}')
    
    for i, (bar, auc) in enumerate(zip(bars, aurocs)):
        ax.text(auc + 0.005, i, f'{auc:.4f}', 
                va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('AUC-ROC', fontsize=14, fontweight='bold')
    ax.set_ylabel('Patologia', fontsize=14, fontweight='bold')
    ax.set_title('Performance do Ensemble por Patologia', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim([0.65, 0.90])
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ensemble_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n Gráfico salvo em: {output_path}")
    plt.close()

def main():
    if not os.path.exists(RESULTS_DIR):
        print(f"Erro: Diretório '{RESULTS_DIR}' não encontrado!")
        print("Execute train_ensemble.py primeiro.")
        return
    
    SEEDS = [42, 85, 123, 777, 999]
    BATCH_SIZE = 64
    WORKERS = 6
    N_LABELS = len(DISEASES)
    DROPOUT_RATE = 0.5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("\nCarregando dataset de teste...")
    
    if not os.path.exists(TEST_DF_PATH):
        print(f"Arquivo de teste não encontrado: {TEST_DF_PATH}")
        print("Gerando split de teste...")
        
        train_df_full = pd.read_csv(TRAIN_DF_PATH)
        if 'path' in train_df_full.columns:
            train_df_full.rename(columns={'path': 'Path'}, inplace=True)
        
        train_df_full = train_df_full.sample(n=100000, random_state=42).reset_index(drop=True)
        stratify_col = (train_df_full['Cardiomegaly'] == 1.0).astype(int)
        
        _, test_df = train_test_split(
            train_df_full, 
            test_size=0.15,
            random_state=42,
            stratify=stratify_col
        )
        test_df = test_df.reset_index(drop=True)
        
        os.makedirs(os.path.dirname(TEST_DF_PATH), exist_ok=True)
        test_df.to_csv(TEST_DF_PATH, index=False)
        print(f" Split de teste salvo em: {TEST_DF_PATH}")
    else:
        test_df = pd.read_csv(TEST_DF_PATH)
        if 'path' in test_df.columns:
            test_df.rename(columns={'path': 'Path'}, inplace=True)
    
    print(f"Total de amostras de teste: {len(test_df):,}")
    
    val_transform = get_transforms()
    test_dataset = CheXpertDataset(test_df, PATH_IMAGE, val_transform, policy='u-ones')
    test_loader = data.DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=WORKERS, 
        pin_memory=True
    )
    
    models_list = []
    
    for seed in SEEDS:
        model_dir = os.path.join(RESULTS_DIR, f"model_seed_{seed}")
        model_path = os.path.join(model_dir, "best_model.pt")
        
        if not os.path.exists(model_path):
            print(f"Modelo não encontrado: {model_path}")
            continue
        
        print(f"Carregando modelo seed {seed}...")
        model = load_model(model_path, device, N_LABELS, DROPOUT_RATE)
        models_list.append((seed, model))
    
    if len(models_list) == 0:
        return
    
    
    print("\nComputando predições individuais...")
    all_model_probs = []
    labels = None
    
    for seed, model in models_list:
        print(f"  Processando seed {seed}")
        curr_labels, curr_probs = compute_predictions(model, test_loader, device)
        
        if labels is None:
            labels = curr_labels
        
        all_model_probs.append(curr_probs)
    
    print("\n Agregando predições do ensemble (média)")
    ensemble_probs = np.mean(all_model_probs, axis=0)
    
    print("\n Calculando AUC-ROC do ensemble")
    aurocs = compute_auroc(labels, ensemble_probs, N_LABELS)
    mean_auroc = np.mean(aurocs)
    
    print(f"\n{'='*60}")
    print(f" AUC-ROC MÉDIO DO ENSEMBLE: {mean_auroc:.4f}")
    print(f"{'='*60}")
    
    print("\n AUC-ROC por patologia:")
    for disease, auc in zip(DISEASES, aurocs):
        print(f"  {disease:30s}: {auc:.4f}")
        
    results_df = pd.DataFrame({
        'Disease': DISEASES,
        'AUC-ROC': aurocs
    })
    results_df = results_df.sort_values('AUC-ROC', ascending=False)
    
    csv_path = os.path.join(RESULTS_DIR, 'ensemble_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Resultados salvos em: {csv_path}")
    
    summary_df = pd.DataFrame({
        'Metric': ['Mean AUC-ROC', 'Std AUC-ROC', 'Min AUC-ROC', 'Max AUC-ROC'],
        'Value': [mean_auroc, np.std(aurocs), np.min(aurocs), np.max(aurocs)]
    })
    
    summary_path = os.path.join(RESULTS_DIR, 'ensemble_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f" Resumo salvo em: {summary_path}")
    
    plot_results(mean_auroc, aurocs, DISEASES, RESULTS_DIR)
    
    print("\nAnálise do ensemble finalizada")

if __name__ == "__main__":
    main()
