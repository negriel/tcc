"""
Análise do Ensemble de 5 Modelos
Carrega modelos treinados e calcula performance combinada

Autor: Gabriel Ribeiro
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm
from PIL import Image

RESULTS_DIR = os.path.join('..', 'results_ensemble')
PATH_IMAGE = r"E:\GabrielRibeiro\chexpert_project\data"
TRAIN_DF_PATH = r"E:\GabrielRibeiro\chexpert_project\data\CheXpert-v1.0-small\train.csv"

DISEASES = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

class DenseNet121_Regularized(nn.Module):
    """DenseNet121 com regularização (Dropout + BatchNorm)."""
    
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

class CheXpertDataset(torch.utils.data.Dataset):
    """Dataset CheXpert com política U-Ones."""
    
    def __init__(self, dataframe, path_image, transform=None, policy='u-ones'):
        self.dataframe = dataframe.reset_index(drop=True)
        self.path_image = path_image
        self.transform = transform
        self.policy = policy
        self.labels_columns = DISEASES
        
    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        caminho_imagem = item["Path"]
        img_path = os.path.join(self.path_image, caminho_imagem).replace("/", os.sep)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Imagem não encontrada: {img_path}")
        
        img = Image.open(img_path).convert("RGB")
        if self.transform:
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

def evaluate_ensemble(models, data_loader, device, num_classes=14):
    """
    Avalia ensemble fazendo média das predições de todos os modelos.
    
    Returns:
        mean_auroc: AUC-ROC médio
        aurocs: AUC de cada doença
        diseases: Nomes das doenças
    """
    all_labels = []
    all_probs_ensemble = []
        
    with torch.no_grad():
        for imgs, labels, _ in tqdm(data_loader, desc="Processando batches"):
            imgs = imgs.to(device, non_blocking=True)
            
            batch_preds = []
            for model in models:
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    outputs = model(imgs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                batch_preds.append(probs)
            
            ensemble_probs = np.mean(batch_preds, axis=0)
            
            all_labels.append(labels.numpy())
            all_probs_ensemble.append(ensemble_probs)
    
    all_labels = np.vstack(all_labels)
    all_probs_ensemble = np.vstack(all_probs_ensemble)
    
    aurocs = []
    for i in range(num_classes):
        unique_labels = np.unique(all_labels[:, i])
        if len(unique_labels) > 1:
            try:
                auc = roc_auc_score(all_labels[:, i], all_probs_ensemble[:, i])
                aurocs.append(auc)
            except:
                aurocs.append(0.5)
        else:
            aurocs.append(0.5)
    
    return np.mean(aurocs), aurocs, DISEASES

def plot_results(mean_auroc, aurocs, diseases, output_dir):
    """Gera gráfico de barras com AUC por doença."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#2ecc71' if auc >= mean_auroc else '#3498db' for auc in aurocs]
    bars = ax.barh(diseases, aurocs, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.axvline(mean_auroc, color='red', linestyle='--', linewidth=2.5, 
               label=f'Média: {mean_auroc:.4f}', zorder=10)
    
    for i, (bar, auc) in enumerate(zip(bars, aurocs)):
        width = bar.get_width()
        label_x_pos = width + 0.01 if width > 0.5 else width + 0.02
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                f'{auc:.3f}', va='center', ha='left', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('AUC-ROC', fontsize=13, fontweight='bold')
    ax.set_title(f'Performance do Ensemble de 5 Modelos por Doença\nAUC-ROC Médio: {mean_auroc:.4f}', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0.5, 1.0)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'ensemble_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nGráfico salvo em: {output_path}")
    plt.close()

def main():
    print("\n" + "="*70)
    print("ANÁLISE DO ENSEMBLE DE 5 MODELOS - CheXpert")
    print("="*70)
    
    BATCH_SIZE = 64
    WORKERS = 6
    RANDOM_SEED = 85
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
    train_df_full = pd.read_csv(TRAIN_DF_PATH)
    
    if 'path' in train_df_full.columns:
        train_df_full.rename(columns={'path': 'Path'}, inplace=True)
    
    train_df_full = train_df_full.sample(n=100000, random_state=RANDOM_SEED).reset_index(drop=True)
    
    stratify_col = (train_df_full['Cardiomegaly'] == 1.0).astype(int)
    _, val_df = train_test_split(
        train_df_full,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=stratify_col
    )
    
    val_df = val_df.reset_index(drop=True)
    print(f"Validação: {len(val_df):,} imagens")
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    val_dataset = CheXpertDataset(val_df, PATH_IMAGE, val_transform, policy='u-ones')
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"{len(val_loader)} batches de validação")
    
    # Carrega os 5 modelos do ensemble
    seeds = [85, 42, 123, 777, 999]
    models = []
    
    print(f"\nCarregando {len(seeds)} modelos do ensemble...")
    
    for seed in seeds:
        model_path = os.path.join(RESULTS_DIR, f"model_seed_{seed}", "best_model.pt")
        
        if not os.path.exists(model_path):
            print(f"ERRO: Modelo seed {seed} não encontrado em {model_path}")
            continue
        
        model = DenseNet121_Regularized(num_classes=14, dropout_rate=0.5)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        auc = checkpoint.get('best_auroc', 'N/A')
        epoch = checkpoint.get('epoch', 'N/A')
        
        auc_str = f"{auc:.4f}" if isinstance(auc, (int, float)) else str(auc)
        epoch_str = f"{epoch}" if isinstance(epoch, int) else str(epoch)
        
        print(f"Seed {seed:3d}: AUC {auc_str} (época {epoch_str})")
        models.append(model)
    
    if len(models) != 5:
        print(f"\nERRO: Apenas {len(models)}/5 modelos carregados!")
        print(f"Verifique se todos os modelos estão em: {RESULTS_DIR}")
        return
    
    print(f"\n{len(models)} modelos carregados com sucesso!")
    
    mean_auroc, aurocs, diseases = evaluate_ensemble(models, val_loader, device)
    
    print("\n" + "="*70)
    print("RESULTADOS DO ENSEMBLE")
    print("="*70)
    print(f"\nAUC-ROC MÉDIO DO ENSEMBLE: {mean_auroc:.4f}\n")
    
    print("AUC-ROC por Doença:")
    print("-" * 60)
    for disease, auc in zip(diseases, aurocs):
        marker = "*" if auc >= mean_auroc else " "
        print(f"{marker} {disease:32s}: {auc:.4f}")
    
    # Salva resultados
    
    results_df = pd.DataFrame({
        'Disease': diseases,
        'AUC-ROC': aurocs,
        'Above_Mean': [auc >= mean_auroc for auc in aurocs]
    })
    csv_path = os.path.join(RESULTS_DIR, 'ensemble_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"CSV salvo: {csv_path}")
    
    plot_results(mean_auroc, aurocs, diseases, RESULTS_DIR)
    
    summary = {
        'Mean_AUC': mean_auroc,
        'Std_AUC': np.std(aurocs),
        'Min_AUC': np.min(aurocs),
        'Max_AUC': np.max(aurocs),
        'Median_AUC': np.median(aurocs),
        'Num_Models': len(models),
        'Num_Diseases': len(diseases)
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(RESULTS_DIR, 'ensemble_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Sumário salvo: {summary_path}")
    
    print("\n" + "="*70)
    print("ANÁLISE COMPARATIVA")
    print("="*70)
    
    individual_aucs = [0.7516, 0.7602, 0.7529, 0.7575, 0.7596]
    mean_individual = np.mean(individual_aucs)
    
    print(f"\nPerformance Individual:")
    print(f"  Média:  {mean_individual:.4f}")
    print(f"  Desvio: {np.std(individual_aucs):.4f}")
    
    print(f"\nPerformance Ensemble:")
    print(f"  AUC:    {mean_auroc:.4f}")
    ganho = mean_auroc - mean_individual
    ganho_pct = 100 * ganho / mean_individual
    print(f"  Ganho:  +{ganho:.4f} ({ganho_pct:.1f}%)")
    
    print(f"\nMelhores Doenças (Top 5):")
    top_5_idx = np.argsort(aurocs)[-5:][::-1]
    for idx in top_5_idx:
        print(f"  {diseases[idx]:32s}: {aurocs[idx]:.4f}")
    
    print(f"\nPiores Doenças (Bottom 5):")
    bottom_5_idx = np.argsort(aurocs)[:5]
    for idx in bottom_5_idx:
        print(f"  {diseases[idx]:32s}: {aurocs[idx]:.4f}")
    
    # Finalização
    print("\n" + "="*70)
    print("ANÁLISE CONCLUÍDA")
    print("="*70)
    print(f"\nResultados salvos em: {RESULTS_DIR}/")
    print("  - ensemble_results.csv")
    print("  - ensemble_summary.csv")
    print("  - ensemble_performance.png")

if __name__ == "__main__":
    main()
