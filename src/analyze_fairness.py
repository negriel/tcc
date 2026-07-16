"""
 Análise de Equidade por Sexo e Faixa Etária
 Dataset: CheXpert (com colunas 'Sex' e 'Age')
 """


 import os
 import torch
 import torch.nn as nn
 import numpy as np
 import pandas as pd
 import matplotlib.pyplot as plt
 import seaborn as sns
 from sklearn.metrics import roc_auc_score
 from sklearn.model_selection import train_test_split
 from torch.utils.data import DataLoader
 import torchvision.transforms as transforms
 from tqdm import tqdm
 from PIL import Image




# =========================================================================
# CONFIGURAÇÕES GLOBAIS
# =========================================================================
RESULTS_DIR = "results_ensemble"
FAIRNESS_DIR = "results_fairness"
PATH_IMAGE = r"E:\GabrielRibeiro\chexpert_project\data"
TRAIN_DF_PATH = r"E:\GabrielRibeiro\chexpert_project\data\CheXpert-v1.0-small\train.csv"


DISEASES = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]


# =========================================================================
# CLASSES DO MODELO E DATASET
# =========================================================================
class DenseNet121_Regularized(nn.Module):
    """DenseNet121 com regularização"""
    def __init__(self, num_classes=14, dropout_rate=0.5):
        super().__init__()
        from torchvision import models
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
    """Dataset CheXpert com política U-Ones"""
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



# =========================================================================
# FUNÇÕES DE ANÁLISE DE EQUIDADE
# =========================================================================
def evaluate_group(models, dataloader, device):
    """
    Avalia ensemble em um subgrupo específico.


    Returns:
          mean_auc (float): AUC médio
          aucs_per_disease (list): AUC de cada doença
    """
    all_labels = []
    all_probs = []


    with torch.no_grad():
          for imgs, labels, _ in dataloader:
              imgs = imgs.to(device, non_blocking=True)


              # Predições de cada modelo
              batch_preds = []
              for model in models:
                  with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                      outputs = model(imgs)
                  probs = torch.sigmoid(outputs).cpu().numpy()
                  batch_preds.append(probs)


              # Ensemble
              ensemble_probs = np.mean(batch_preds, axis=0)




              all_labels.append(labels.numpy())
              all_probs.append(ensemble_probs)


    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)


    # AUC por doença
    aucs = []
    for i in range(len(DISEASES)):
          if len(np.unique(all_labels[:, i])) > 1:
              try:
                  auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                  aucs.append(auc)
              except:
                  aucs.append(np.nan)
          else:
              aucs.append(np.nan)


    return np.nanmean(aucs), aucs



def analyze_by_sex(models, val_df, path_image, device, output_dir):
    """
    Analisa disparidade de performance por sexo.
    """
    print("\n" + "="*70)
    print("      ANÁLISE DE EQUIDADE POR SEXO")
    print("="*70)


    # Preparação
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          normalize
    ])


    results = {}


    # Verifica se coluna 'Sex' existe
    if 'Sex' not in val_df.columns:
          print(" ERRO: Coluna 'Sex' não encontrada no dataset!")
          print(f"    Colunas disponíveis: {val_df.columns.tolist()}")
          return results


    # Mapeamento de valores possíveis
    sex_mapping = {
          'Male': ['Male', 'M', 'male', 'm', 1, 1.0],
          'Female': ['Female', 'F', 'female', 'f', 0, 0.0]




   }


   for sex_label, sex_values in sex_mapping.items():
       # Filtra por sexo
       df_sex = val_df[val_df['Sex'].isin(sex_values)].copy().reset_index(drop=True)


       if len(df_sex) == 0:
           print(f"\n     Nenhum dado encontrado para sexo '{sex_label}'")
           continue


       print(f"\n {sex_label}:")
       print(f"     • Amostras: {len(df_sex):,}")


       # Cria dataset e loader
       dataset = CheXpertDataset(df_sex, path_image, val_transform, policy='u-ones')
       loader = DataLoader(dataset, batch_size=64, shuffle=False,
                            num_workers=4, pin_memory=True)


       # Avalia
       mean_auc, aucs = evaluate_group(models, loader, device)


       results[sex_label] = {
           'mean_auc': mean_auc,
           'aucs': aucs,
           'n_samples': len(df_sex)
       }


       print(f"     • AUC-ROC: {mean_auc:.4f}")


   # Calcula gaps
   if 'Male' in results and 'Female' in results:
       gap_mean = abs(results['Male']['mean_auc'] - results['Female']['mean_auc'])
       gap_pct = (gap_mean / min(results['Male']['mean_auc'],
results['Female']['mean_auc'])) * 100


       print(f"\n DISPARIDADE:")
       print(f"     • Gap absoluto: {gap_mean:.4f}")
       print(f"     • Gap relativo: {gap_pct:.1f}%")


       if gap_mean < 0.01:
           print("      Disparidade BAIXA (< 1%)")
       elif gap_mean < 0.03:
           print("        Disparidade MODERADA (1-3%)")
       else:
           print("      Disparidade ALTA (> 3%)")


       # Gráfico comparativo
       plot_sex_comparison(results, output_dir)


   return results




def analyze_by_age(models, val_df, path_image, device, output_dir):
    """
    Analisa disparidade de performance por faixa etária.
    """
    print("\n" + "="*70)
    print("     ANÁLISE DE EQUIDADE POR FAIXA ETÁRIA")
    print("="*70)


    # Preparação
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          normalize
    ])


    results = {}


    # Verifica se coluna 'Age' existe
    if 'Age' not in val_df.columns:
          print(" ERRO: Coluna 'Age' não encontrada no dataset!")
          print(f"    Colunas disponíveis: {val_df.columns.tolist()}")
          return results


    # Remove NaNs de idade
    val_df_clean = val_df.dropna(subset=['Age']).copy()
    print(f" Total de amostras com idade válida: {len(val_df_clean):,}")


    # Define faixas etárias
    val_df_clean['age_group'] = pd.cut(
          val_df_clean['Age'],
          bins=[0, 40, 65, 120],
          labels=['< 40 anos', '40-65 anos', '> 65 anos']
    )


    for age_group in ['< 40 anos', '40-65 anos', '> 65 anos']:
          df_age = val_df_clean[val_df_clean['age_group'] ==
age_group].copy().reset_index(drop=True)


          if len(df_age) == 0:
              print(f"\n    Nenhum dado encontrado para '{age_group}'")
              continue


          print(f"\n {age_group}:")
          print(f"    • Amostras: {len(df_age):,}")
          print(f"    • Idade média: {df_age['Age'].mean():.1f} anos")


          # Cria dataset e loader
          dataset = CheXpertDataset(df_age, path_image, val_transform, policy='u-ones')
          loader = DataLoader(dataset, batch_size=64, shuffle=False,




                             num_workers=4, pin_memory=True)


         # Avalia
         mean_auc, aucs = evaluate_group(models, loader, device)


         results[age_group] = {
             'mean_auc': mean_auc,
             'aucs': aucs,
             'n_samples': len(df_age)
         }


         print(f"    • AUC-ROC: {mean_auc:.4f}")


   # Análise de variância
   if len(results) > 1:
         aucs_list = [r['mean_auc'] for r in results.values()]
         variance = np.var(aucs_list)
         range_auc = max(aucs_list) - min(aucs_list)


         print(f"\n VARIAÇÃO ENTRE FAIXAS ETÁRIAS:")
         print(f"    • Variância: {variance:.6f}")
         print(f"    • Range: {range_auc:.4f}")


         # Gráfico
         plot_age_comparison(results, output_dir)


   return results



def plot_sex_comparison(results, output_dir):
   """
   Gera gráfico comparativo de performance por sexo.
   """
   if 'Male' not in results or 'Female' not in results:
         return


   male_aucs = results['Male']['aucs']
   female_aucs = results['Female']['aucs']


   # Remove NaNs
   valid_idx = [i for i in range(len(DISEASES)) if not np.isnan(male_aucs[i]) and not
np.isnan(female_aucs[i])]
   diseases_valid = [DISEASES[i] for i in valid_idx]
   male_aucs_valid = [male_aucs[i] for i in valid_idx]
   female_aucs_valid = [female_aucs[i] for i in valid_idx]


   # Gráfico
   fig, ax = plt.subplots(figsize=(14, 8))


   x = np.arange(len(diseases_valid))
   width = 0.35




   bars1 = ax.bar(x - width/2, male_aucs_valid, width, label='Masculino',
                     color='steelblue', edgecolor='black', linewidth=0.5)
   bars2 = ax.bar(x + width/2, female_aucs_valid, width, label='Feminino',
                     color='coral', edgecolor='black', linewidth=0.5)


   ax.set_xlabel('Patologia', fontsize=13, fontweight='bold')
   ax.set_ylabel('AUC-ROC', fontsize=13, fontweight='bold')
   ax.set_title('Comparação de Performance por Sexo', fontsize=15, fontweight='bold')
   ax.set_xticks(x)
   ax.set_xticklabels(diseases_valid, rotation=45, ha='right')
   ax.legend(fontsize=11)
   ax.grid(axis='y', alpha=0.3)
   ax.set_ylim(0.5, 1.0)


   # Anotações
   for i, (m, f) in enumerate(zip(male_aucs_valid, female_aucs_valid)):
         gap = abs(m - f)
         if gap > 0.02:   # Destaca gaps > 2%
             mid_point = (m + f) / 2
             ax.plot([i - width/2, i + width/2], [m, f], 'r--', linewidth=1.5)
             ax.text(i, mid_point, f'” {gap:.3f}', ha='center', va='bottom',
                    fontsize=8, color='red', fontweight='bold')


   plt.tight_layout()
   output_path = os.path.join(output_dir, 'fairness_sex_comparison.png')
   plt.savefig(output_path, dpi=300, bbox_inches='tight')
   print(f"\n Gráfico salvo: {output_path}")
   plt.close()



def plot_age_comparison(results, output_dir):
   """
   Gera gráfico comparativo de performance por faixa etária.
   """
   age_groups = list(results.keys())
   mean_aucs = [results[ag]['mean_auc'] for ag in age_groups]
   n_samples = [results[ag]['n_samples'] for ag in age_groups]


   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))


   # Gráfico 1: AUC por faixa etária
   colors = ['#3498db', '#2ecc71', '#e74c3c']
   bars = ax1.bar(age_groups, mean_aucs, color=colors, edgecolor='black', linewidth=1)
   ax1.set_ylabel('AUC-ROC Médio', fontsize=12, fontweight='bold')
   ax1.set_title('Performance por Faixa Etária', fontsize=14, fontweight='bold')
   ax1.set_ylim(0.5, 1.0)
   ax1.grid(axis='y', alpha=0.3)


   # Anotações
   for bar, auc in zip(bars, mean_aucs):
         ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')




    # Gráfico 2: Distribuição de amostras
    ax2.bar(age_groups, n_samples, color=colors, edgecolor='black', linewidth=1, alpha=0.7)
    ax2.set_ylabel('Número de Amostras', fontsize=12, fontweight='bold')
    ax2.set_title('Distribuição de Amostras por Faixa Etária', fontsize=14,
fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)


    for i, (ag, n) in enumerate(zip(age_groups, n_samples)):
        ax2.text(i, n + max(n_samples)*0.02, f'{n:,}', ha='center', va='bottom',
fontweight='bold')


    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fairness_age_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Gráfico salvo: {output_path}")
    plt.close()



# =========================================================================
# PIPELINE PRINCIPAL
# =========================================================================
def main():
    os.makedirs(FAIRNESS_DIR, exist_ok=True)


    print("\n" + "="*70)
    print(" ANÁLISE DE EQUIDADE - ENSEMBLE CheXpert")
    print("="*70)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n Dispositivo: {device}")


    # -------------------------------------------------------------------------
    # 1. CARREGA DADOS DE VALIDAÇÃO
    # -------------------------------------------------------------------------
    print("\n Carregando dados de validação...")
    train_df_full = pd.read_csv(TRAIN_DF_PATH)


    # Padronização
    if 'path' in train_df_full.columns:
        train_df_full.rename(columns={'path': 'Path'}, inplace=True)


    # Mesmo subset do treino
    train_df_full = train_df_full.sample(n=100000, random_state=85).reset_index(drop=True)


    # Split
    stratify_col = (train_df_full['Cardiomegaly'] == 1.0).astype(int)
    _, val_df = train_test_split(
        train_df_full,
        test_size=0.15,
        random_state=85,
        stratify=stratify_col




)


val_df = val_df.reset_index(drop=True)


print(f"      Validação: {len(val_df):,} imagens")
print(f"\n Colunas disponíveis no dataset:")
print(f"     {val_df.columns.tolist()}")


# Estatísticas demográficas
if 'Sex' in val_df.columns:
    print(f"\n Distribuição por Sexo:")
    print(val_df['Sex'].value_counts())


if 'Age' in val_df.columns:
    print(f"\n Estatísticas de Idade:")
    print(f"     • Média: {val_df['Age'].mean():.1f} anos")
    print(f"     • Mediana: {val_df['Age'].median():.1f} anos")
    print(f"     • Range: {val_df['Age'].min():.0f}-{val_df['Age'].max():.0f} anos")


# -------------------------------------------------------------------------
# 2. CARREGA MODELOS DO ENSEMBLE
# -------------------------------------------------------------------------
seeds = [85, 42, 123, 777, 999]
models = []


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
    models.append(model)


if len(models) != 5:
    print(f"\n ERRO: Apenas {len(models)}/5 modelos carregados!")
    return


print(f"\n {len(models)} modelos carregados com sucesso!")


# -------------------------------------------------------------------------
# 3. ANÁLISE POR SEXO
# -------------------------------------------------------------------------




sex_results = analyze_by_sex(models, val_df, PATH_IMAGE, device, FAIRNESS_DIR)


# -------------------------------------------------------------------------
# 4. ANÁLISE POR IDADE
# -------------------------------------------------------------------------
age_results = analyze_by_age(models, val_df, PATH_IMAGE, device, FAIRNESS_DIR)


# -------------------------------------------------------------------------
# 5. SALVA RESULTADOS
# -------------------------------------------------------------------------
print("\n Salvando resultados...")


# CSV Sexo
if sex_results:
    sex_df = pd.DataFrame({
          'Group': list(sex_results.keys()),
          'Mean_AUC': [r['mean_auc'] for r in sex_results.values()],
          'N_Samples': [r['n_samples'] for r in sex_results.values()]
    })
    sex_df.to_csv(os.path.join(FAIRNESS_DIR, 'fairness_sex_results.csv'), index=False)
    print(f"      {FAIRNESS_DIR}/fairness_sex_results.csv")


# CSV Idade
if age_results:
    age_df = pd.DataFrame({
          'Group': list(age_results.keys()),
          'Mean_AUC': [r['mean_auc'] for r in age_results.values()],
          'N_Samples': [r['n_samples'] for r in age_results.values()]
    })
    age_df.to_csv(os.path.join(FAIRNESS_DIR, 'fairness_age_results.csv'), index=False)
    print(f"      {FAIRNESS_DIR}/fairness_age_results.csv")


print(" ANÁLISE DE EQUIDADE CONCLUÍDA!")
print(f"\n Resultados salvos em: {FAIRNESS_DIR}/")
print("\n Análise completa!")




 if __name__ == "__main__":
       main()
