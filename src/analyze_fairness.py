"""
Análise de Equidade por Sexo e Faixa Etária — CheXpert (colunas 'Sex' e 'Age').

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
from PIL import Image

RESULTS_DIR = os.path.join('..', 'results_ensemble')
FAIRNESS_DIR = os.path.join('..', 'results_fairness')
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


def get_val_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])


def evaluate_group(models_list, dataloader, device):
    """Avalia o ensemble num subgrupo. Retorna (AUC-14 medio, lista de AUCs por doenca)."""
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels, _ in dataloader:
            imgs = imgs.to(device, non_blocking=True)

            batch_preds = []
            for model in models_list:
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    outputs = model(imgs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                batch_preds.append(probs)

            ensemble_probs = np.mean(batch_preds, axis=0)
            all_labels.append(labels.numpy())
            all_probs.append(ensemble_probs)

    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)

    aucs = []
    for i in range(len(DISEASES)):
        if len(np.unique(all_labels[:, i])) > 1:
            try:
                aucs.append(roc_auc_score(all_labels[:, i], all_probs[:, i]))
            except ValueError:
                aucs.append(np.nan)
        else:
            aucs.append(np.nan)

    return np.nanmean(aucs), aucs


def auc5_from(aucs):
    """AUC medio das 5 patologias da competition task, ignorando NaN."""
    idx_5 = [DISEASES.index(p) for p in COMPETITION_PATHOLOGIES]
    vals = [aucs[i] for i in idx_5 if not np.isnan(aucs[i])]
    return np.mean(vals) if vals else np.nan


def analyze_by_sex(models_list, test_df, path_image, device, output_dir):
    print("\n" + "=" * 70)
    print("      ANÁLISE DE EQUIDADE POR SEXO")
    print("=" * 70)

    val_transform = get_val_transform()
    results = {}

    if 'Sex' not in test_df.columns:
        print(" ERRO: Coluna 'Sex' não encontrada no dataset!")
        print(f"    Colunas disponíveis: {test_df.columns.tolist()}")
        return results

    sex_mapping = {
        'Male': ['Male', 'M', 'male', 'm', 1, 1.0],
        'Female': ['Female', 'F', 'female', 'f', 0, 0.0],
    }

    for sex_label, sex_values in sex_mapping.items():
        df_sex = test_df[test_df['Sex'].isin(sex_values)].copy().reset_index(drop=True)

        if len(df_sex) == 0:
            print(f"\n     Nenhum dado encontrado para sexo '{sex_label}'")
            continue

        print(f"\n {sex_label}:")
        print(f"     - Amostras: {len(df_sex):,}")

        dataset = CheXpertDataset(df_sex, path_image, val_transform, policy='u-ones')
        loader = DataLoader(dataset, batch_size=64, shuffle=False,
                            num_workers=4, pin_memory=True)

        mean_auc, aucs = evaluate_group(models_list, loader, device)
        results[sex_label] = {
            'mean_auc': mean_auc,
            'auc5': auc5_from(aucs),
            'aucs': aucs,
            'n_samples': len(df_sex),
        }
        print(f"     - AUC-ROC (14): {mean_auc:.4f} | AUC-ROC (5): {results[sex_label]['auc5']:.4f}")

    if 'Male' in results and 'Female' in results:
        gap_mean = abs(results['Male']['mean_auc'] - results['Female']['mean_auc'])
        gap_pct = (gap_mean / min(results['Male']['mean_auc'],
                                  results['Female']['mean_auc'])) * 100

        print(f"\n DISPARIDADE (sexo):")
        print(f"     - Gap absoluto: {gap_mean:.4f}")
        print(f"     - Gap relativo: {gap_pct:.1f}%")

        if gap_mean < 0.01:
            print("     Disparidade BAIXA (< 1%)")
        elif gap_mean < 0.03:
            print("     Disparidade MODERADA (1-3%)")
        else:
            print("     Disparidade ALTA (> 3%)")

        plot_sex_comparison(results, output_dir)

    return results


def analyze_by_age(models_list, test_df, path_image, device, output_dir):
    print("\n" + "=" * 70)
    print("     ANÁLISE DE EQUIDADE POR FAIXA ETÁRIA")
    print("=" * 70)

    val_transform = get_val_transform()
    results = {}

    if 'Age' not in test_df.columns:
        print(" ERRO: Coluna 'Age' não encontrada no dataset!")
        print(f"    Colunas disponíveis: {test_df.columns.tolist()}")
        return results

    test_df_clean = test_df.dropna(subset=['Age']).copy()
    print(f" Total de amostras com idade válida: {len(test_df_clean):,}")

    test_df_clean['age_group'] = pd.cut(
        test_df_clean['Age'],
        bins=[0, 40, 65, 120],
        labels=['< 40 anos', '40-65 anos', '> 65 anos']
    )

    for age_group in ['< 40 anos', '40-65 anos', '> 65 anos']:
        df_age = test_df_clean[test_df_clean['age_group'] == age_group].copy().reset_index(drop=True)

        if len(df_age) == 0:
            print(f"\n    Nenhum dado encontrado para '{age_group}'")
            continue

        print(f"\n {age_group}:")
        print(f"    - Amostras: {len(df_age):,}")
        print(f"    - Idade média: {df_age['Age'].mean():.1f} anos")

        dataset = CheXpertDataset(df_age, path_image, val_transform, policy='u-ones')
        loader = DataLoader(dataset, batch_size=64, shuffle=False,
                            num_workers=4, pin_memory=True)

        mean_auc, aucs = evaluate_group(models_list, loader, device)
        results[age_group] = {
            'mean_auc': mean_auc,
            'auc5': auc5_from(aucs),
            'aucs': aucs,
            'n_samples': len(df_age),
            'mean_age': df_age['Age'].mean(),
        }
        print(f"    - AUC-ROC (14): {mean_auc:.4f} | AUC-ROC (5): {results[age_group]['auc5']:.4f}")

    if len(results) > 1:
        aucs_list = [r['mean_auc'] for r in results.values()]
        variance = np.var(aucs_list)
        range_auc = max(aucs_list) - min(aucs_list)
        print(f"\n VARIAÇÃO ENTRE FAIXAS ETÁRIAS:")
        print(f"    - Variância: {variance:.6f}")
        print(f"    - Range (gap entre extremos): {range_auc:.4f}")
        plot_age_comparison(results, output_dir)

    return results


def plot_sex_comparison(results, output_dir):
    if 'Male' not in results or 'Female' not in results:
        return

    male_aucs = results['Male']['aucs']
    female_aucs = results['Female']['aucs']

    valid_idx = [i for i in range(len(DISEASES))
                 if not np.isnan(male_aucs[i]) and not np.isnan(female_aucs[i])]
    diseases_valid = [DISEASES[i] for i in valid_idx]
    male_valid = [male_aucs[i] for i in valid_idx]
    female_valid = [female_aucs[i] for i in valid_idx]

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(diseases_valid))
    width = 0.35

    ax.bar(x - width / 2, male_valid, width, label='Masculino',
           color='steelblue', edgecolor='black', linewidth=0.5)
    ax.bar(x + width / 2, female_valid, width, label='Feminino',
           color='coral', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Patologia', fontsize=13, fontweight='bold')
    ax.set_ylabel('AUC-ROC', fontsize=13, fontweight='bold')
    ax.set_title('Comparação de Performance por Sexo', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(diseases_valid, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0.5, 1.0)

    for i, (m, f) in enumerate(zip(male_valid, female_valid)):
        gap = abs(m - f)
        if gap > 0.02:
            mid_point = (m + f) / 2
            ax.plot([i - width / 2, i + width / 2], [m, f], 'r--', linewidth=1.5)
            ax.text(i, mid_point, f'Δ {gap:.3f}', ha='center', va='bottom',
                    fontsize=8, color='red', fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fairness_sex_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n Gráfico salvo: {output_path}")
    plt.close()


def plot_age_comparison(results, output_dir):
    age_groups = list(results.keys())
    mean_aucs = [results[ag]['mean_auc'] for ag in age_groups]
    n_samples = [results[ag]['n_samples'] for ag in age_groups]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax1.bar(age_groups, mean_aucs, color=colors, edgecolor='black', linewidth=1)
    ax1.set_ylabel('AUC-ROC Médio', fontsize=12, fontweight='bold')
    ax1.set_title('Performance por Faixa Etária', fontsize=14, fontweight='bold')
    ax1.set_ylim(0.5, 1.0)
    ax1.grid(axis='y', alpha=0.3)

    for bar, auc in zip(bars, mean_aucs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')

    ax2.bar(age_groups, n_samples, color=colors, edgecolor='black', linewidth=1, alpha=0.7)
    ax2.set_ylabel('Número de Amostras', fontsize=12, fontweight='bold')
    ax2.set_title('Distribuição de Amostras por Faixa Etária', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    for i, (ag, n) in enumerate(zip(age_groups, n_samples)):
        ax2.text(i, n + max(n_samples) * 0.02, f'{n:,}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fairness_age_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Gráfico salvo: {output_path}")
    plt.close()


def load_common_test_df():
    if not os.path.exists(COMMON_TEST_PATH):
        raise FileNotFoundError(
            f"Conjunto de teste comum nao encontrado em {COMMON_TEST_PATH}. "
            f"Rode prepare_split.py primeiro."
        )
    full = pd.read_csv(TRAIN_DF_PATH)
    if 'path' in full.columns:
        full.rename(columns={'path': 'Path'}, inplace=True)
    test_paths = pd.read_csv(COMMON_TEST_PATH)['Path'].tolist()
    return full[full['Path'].isin(test_paths)].reset_index(drop=True)


def main():
    os.makedirs(FAIRNESS_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print(" ANÁLISE DE EQUIDADE - ENSEMBLE CheXpert")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n Dispositivo: {device}")

    print("\n Carregando conjunto de teste comum...")
    test_df = load_common_test_df()
    print(f"      Teste comum: {len(test_df):,} imagens")
    print(f"\n Colunas disponíveis: {test_df.columns.tolist()}")

    if 'Sex' in test_df.columns:
        print(f"\n Distribuição por Sexo:")
        print(test_df['Sex'].value_counts())

    if 'Age' in test_df.columns:
        print(f"\n Estatísticas de Idade:")
        print(f"     - Média: {test_df['Age'].mean():.1f} anos")
        print(f"     - Mediana: {test_df['Age'].median():.1f} anos")
        print(f"     - Range: {test_df['Age'].min():.0f}-{test_df['Age'].max():.0f} anos")

    seeds = [85, 42, 123, 777, 999]
    models_list = []
    print(f"\n Carregando {len(seeds)} modelos do ensemble...")

    for seed in seeds:
        model_path = os.path.join(RESULTS_DIR, f"model_seed_{seed}", "best_model.pt")
        if not os.path.exists(model_path):
            print(f"      Modelo seed {seed} não encontrado!")
            continue

        model = DenseNet121_Regularized(num_classes=14, dropout_rate=0.5)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"      Modelo seed {seed} carregado")
        models_list.append(model)

    if len(models_list) != 5:
        print(f"\n ERRO: Apenas {len(models_list)}/5 modelos carregados!")
        return

    print(f"\n {len(models_list)} modelos carregados com sucesso!")

    sex_results = analyze_by_sex(models_list, test_df, PATH_IMAGE, device, FAIRNESS_DIR)
    age_results = analyze_by_age(models_list, test_df, PATH_IMAGE, device, FAIRNESS_DIR)

    print("\n Salvando resultados...")

    if sex_results:
        sex_df = pd.DataFrame({
            'Group': list(sex_results.keys()),
            'Mean_AUC_14': [r['mean_auc'] for r in sex_results.values()],
            'Mean_AUC_5': [r['auc5'] for r in sex_results.values()],
            'N_Samples': [r['n_samples'] for r in sex_results.values()],
        })
        sex_df.to_csv(os.path.join(FAIRNESS_DIR, 'fairness_sex_results.csv'), index=False)
        print(f"      {FAIRNESS_DIR}/fairness_sex_results.csv")

    if age_results:
        age_df = pd.DataFrame({
            'Group': list(age_results.keys()),
            'Mean_AUC_14': [r['mean_auc'] for r in age_results.values()],
            'Mean_AUC_5': [r['auc5'] for r in age_results.values()],
            'Mean_Age': [r['mean_age'] for r in age_results.values()],
            'N_Samples': [r['n_samples'] for r in age_results.values()],
        })
        age_df.to_csv(os.path.join(FAIRNESS_DIR, 'fairness_age_results.csv'), index=False)
        print(f"      {FAIRNESS_DIR}/fairness_age_results.csv")

    print("\n ANÁLISE DE EQUIDADE CONCLUÍDA!")
    print(f" Resultados salvos em: {FAIRNESS_DIR}/")


if __name__ == "__main__":
    main()