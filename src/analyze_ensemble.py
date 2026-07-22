"""
Análise do ensemble de 5 modelos: carrega os checkpoints treinados e calcula
a performance combinada no CONJUNTO DE TESTE COMUM (common_test_set.csv).
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image

RESULTS_DIR = os.path.join('..', 'results_ensemble')
PATH_IMAGE = r"E:\GabrielRibeiro\chexpert_project\data"
TRAIN_DF_PATH = r"E:\GabrielRibeiro\chexpert_project\data\CheXpert-v1.0-small\train.csv"
COMMON_TEST_PATH = os.path.join(RESULTS_DIR, 'common_test_set.csv')

DISEASES = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

COMPETITION_PATHOLOGIES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']


class DenseNet121_Regularized(nn.Module):
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


def load_common_test_df():
    """Carrega o conjunto de teste comum e junta com os rotulos do train.csv."""
    if not os.path.exists(COMMON_TEST_PATH):
        raise FileNotFoundError(
            f"Conjunto de teste comum nao encontrado em {COMMON_TEST_PATH}. "
            f"Rode prepare_split.py primeiro."
        )

    full = pd.read_csv(TRAIN_DF_PATH)
    if 'path' in full.columns:
        full.rename(columns={'path': 'Path'}, inplace=True)

    test_paths = pd.read_csv(COMMON_TEST_PATH)['Path'].tolist()
    test_df = full[full['Path'].isin(test_paths)].reset_index(drop=True)
    return test_df


def evaluate_single(model, data_loader, device, num_classes=14):
    """Avalia UM modelo isolado. Retorna (AUC-14 medio, AUC-5 medio, lista de AUCs)."""
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels, _ in data_loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_labels.append(labels.numpy())
            all_probs.append(probs)

    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)

    aucs = []
    for i in range(num_classes):
        if len(np.unique(all_labels[:, i])) > 1:
            try:
                aucs.append(roc_auc_score(all_labels[:, i], all_probs[:, i]))
            except ValueError:
                aucs.append(0.5)
        else:
            aucs.append(0.5)

    idx_5 = [DISEASES.index(p) for p in COMPETITION_PATHOLOGIES]
    auc5 = np.mean([aucs[i] for i in idx_5])
    return np.mean(aucs), auc5, aucs


def evaluate_ensemble(models, data_loader, device, num_classes=14):
    """Avalia o ensemble fazendo a média das predições de todos os modelos."""
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

    aucs = []
    for i in range(num_classes):
        if len(np.unique(all_labels[:, i])) > 1:
            try:
                aucs.append(roc_auc_score(all_labels[:, i], all_probs_ensemble[:, i]))
            except ValueError:
                aucs.append(0.5)
        else:
            aucs.append(0.5)

    return np.mean(aucs), aucs, DISEASES


def plot_results(mean_auroc, aurocs, diseases, output_dir):
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = ['#2ecc71' if auc >= mean_auroc else '#3498db' for auc in aurocs]
    bars = ax.barh(diseases, aurocs, color=colors, edgecolor='black', linewidth=0.5)

    ax.axvline(mean_auroc, color='red', linestyle='--', linewidth=2.5,
               label=f'Média: {mean_auroc:.4f}', zorder=10)

    for bar, auc in zip(bars, aurocs):
        width = bar.get_width()
        label_x_pos = width + 0.01 if width > 0.5 else width + 0.02
        ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2,
                f'{auc:.3f}', va='center', ha='left', fontsize=9, fontweight='bold')

    ax.set_xlabel('AUC-ROC', fontsize=13, fontweight='bold')
    ax.set_title(f'Performance do Ensemble de 5 Modelos por Doença\nAUC-ROC Médio (14 patologias): {mean_auroc:.4f}',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0.5, 1.0)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ensemble_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    BATCH_SIZE = 64
    WORKERS = 6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    test_df = load_common_test_df()
    print(f"Conjunto de teste comum: {len(test_df):,} imagens")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    test_dataset = CheXpertDataset(test_df, PATH_IMAGE, val_transform, policy='u-ones')
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    seeds = [85, 42, 123, 777, 999]
    models_list = []
    per_model_rows = []  # AUCs de cada modelo isolado, medidos NO TESTE COMUM

    for seed in seeds:
        model_path = os.path.join(RESULTS_DIR, f"model_seed_{seed}", "best_model.pt")
        if not os.path.exists(model_path):
            print(f"Modelo seed {seed} não encontrado em {model_path}")
            continue

        model = DenseNet121_Regularized(num_classes=14, dropout_rate=0.5)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        models_list.append(model)

        epoch = checkpoint.get('epoch', 'N/A')
        val_auc = checkpoint.get('best_auroc', float('nan'))  # do split interno

        # Avalia o modelo isolado NO TESTE COMUM, para comparacao justa com o ensemble.
        single_auc14, single_auc5, _ = evaluate_single(model, test_loader, device)
        per_model_rows.append({
            'Seed': seed,
            'Epoch': epoch,
            'AUC_14_val_interna': val_auc,   # referencia de convergencia
            'AUC_14_teste_comum': single_auc14,
            'AUC_5_teste_comum': single_auc5,
        })
        print(f"Seed {seed:3d}: teste comum AUC-14 {single_auc14:.4f} | "
              f"AUC-5 {single_auc5:.4f} (época {epoch})")

    if len(models_list) != 5:
        print(f"Apenas {len(models_list)}/5 modelos carregados. Verifique {RESULTS_DIR}")
        return

    mean_auroc, aurocs, diseases = evaluate_ensemble(models_list, test_loader, device)

    # AUC-5: subconjunto da competition task, para comparacao com a literatura.
    idx_5 = [DISEASES.index(p) for p in COMPETITION_PATHOLOGIES]
    auc5 = np.mean([aurocs[i] for i in idx_5])

    print(f"\nAUC-ROC médio do ensemble (14 patologias): {mean_auroc:.4f}")
    print(f"AUC-ROC médio do ensemble (5 patologias, competition task): {auc5:.4f}")
    for disease, auc in zip(diseases, aurocs):
        marker = "*" if disease in COMPETITION_PATHOLOGIES else " "
        print(f"{marker} {disease:32s}: {auc:.4f}")

    results_df = pd.DataFrame({
        'Disease': diseases,
        'AUC-ROC': aurocs,
        'Competition_Task': [d in COMPETITION_PATHOLOGIES for d in diseases]
    })
    results_df.to_csv(os.path.join(RESULTS_DIR, 'ensemble_results.csv'), index=False)

    plot_results(mean_auroc, aurocs, diseases, RESULTS_DIR)

    # Tabela por modelo individual (AUCs no teste comum) — base da Tabela 4/5 do TCC.
    per_model_df = pd.DataFrame(per_model_rows)
    per_model_df.to_csv(os.path.join(RESULTS_DIR, 'individual_results.csv'), index=False)

    # Comparacao individual vs ensemble, agora TODA medida no teste comum.
    ind_14 = per_model_df['AUC_14_teste_comum'].values
    mean_ind_14 = float(np.mean(ind_14))
    best_ind_14 = float(np.max(ind_14))
    best_seed = int(per_model_df.loc[per_model_df['AUC_14_teste_comum'].idxmax(), 'Seed'])
    ganho_sobre_media = mean_auroc - mean_ind_14
    ganho_sobre_melhor = mean_auroc - best_ind_14

    summary_df = pd.DataFrame([{
        'Mean_AUC_14_ensemble': mean_auroc,
        'Mean_AUC_5_ensemble': auc5,
        'Mean_AUC_14_individual': mean_ind_14,
        'Best_AUC_14_individual': best_ind_14,
        'Best_seed': best_seed,
        'Std_AUC_14_individual': float(np.std(ind_14)),
        'Ganho_sobre_media': ganho_sobre_media,
        'Ganho_sobre_melhor': ganho_sobre_melhor,
        'Num_Models': len(models_list),
    }])
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'ensemble_summary.csv'), index=False)

    print(f"\n--- Comparacao (tudo no teste comum) ---")
    print(f"Média individual (AUC-14): {mean_ind_14:.4f} (desvio {np.std(ind_14):.4f})")
    print(f"Melhor individual (AUC-14): {best_ind_14:.4f} (seed {best_seed})")
    print(f"Ensemble (AUC-14): {mean_auroc:.4f}")
    print(f"Ganho do ensemble sobre a média individual: {ganho_sobre_media:+.4f} "
          f"({100 * ganho_sobre_media / mean_ind_14:+.1f}%)")
    print(f"Ganho do ensemble sobre o melhor individual: {ganho_sobre_melhor:+.4f}")
    print(f"\nMelhor seed para quantizar: {best_seed} "
          f"(ajuste BEST_SEED em quantize_model.py se necessario)")

    print(f"\nResultados salvos em: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()