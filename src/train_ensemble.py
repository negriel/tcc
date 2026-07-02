"""
Ensemble Training & Evaluation
Treina múltiplos modelos com seeds diferentes e combina predições

Autor: Gabriel Ribeiro
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from train_finetuning import (
    train_model, 
    DenseNet121_Regularized, 
    CheXpertDataset, 
    get_transforms,
    DISEASES,
    TRAIN_DF_PATH,
    TEST_DF_PATH,
    PATH_IMAGE
)

ENSEMBLE_SEEDS = [85, 42, 123, 777, 999]
ENSEMBLE_DIR = os.path.join('..', 'results_ensemble')
DEBUG_MODE = False

def train_ensemble_models(seeds, base_dir, debug=False):
    """
    Treina múltiplos modelos com seeds diferentes.
    
    Args:
        seeds: Lista de seeds para diversidade
        base_dir: Diretório base para salvar modelos
        debug: Modo debug (treino rápido)
    
    Returns:
        Lista de dicionários com resultados de cada modelo
    """
    os.makedirs(base_dir, exist_ok=True)
    results = []
    
    print(f"\nTREINANDO ENSEMBLE COM {len(seeds)} MODELOS")
    print(f"Seeds: {seeds}")
    print(f"Diretório: {base_dir}\n")
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n{'='*70}")
        print(f"MODELO {i}/{len(seeds)} - SEED {seed}")
        print(f"{'='*70}\n")
        
        model_dir = os.path.join(base_dir, f"model_seed_{seed}")
        
        try:
            result = train_model(
                random_seed=seed,
                results_dir=model_dir,
                debug_mode=debug
            )
            results.append(result)
            print(f"\nModelo {i} finalizado: AUC {result['best_auroc']:.4f}")
            
        except Exception as e:
            print(f"\nErro ao treinar modelo {i} (seed {seed}): {str(e)}")
            continue
    
    print(f"\n{'='*70}")
    print("SUMÁRIO DO ENSEMBLE")
    print(f"{'='*70}")
    for i, res in enumerate(results, 1):
        print(f"Modelo {i} (seed {res['random_seed']}): AUC {res['best_auroc']:.4f}")
    
    if results:
        aucs = [r['best_auroc'] for r in results]
        print(f"\nEstatísticas:")
        print(f"  Média:  {np.mean(aucs):.4f}")
        print(f"  Desvio: {np.std(aucs):.4f}")
        print(f"  Mínimo: {np.min(aucs):.4f}")
        print(f"  Máximo: {np.max(aucs):.4f}")
    
    return results

def load_ensemble_models(model_paths, device):
    """
    Carrega múltiplos modelos treinados.
    
    Args:
        model_paths: Lista de caminhos dos checkpoints
        device: Dispositivo (cuda/cpu)
    
    Returns:
        Lista de modelos carregados em eval mode
    """
    models = []    
    
    for i, path in enumerate(model_paths, 1):
        if not os.path.exists(path):
            print(f"Modelo {i} não encontrado: {path}")
            continue
        
        model = DenseNet121_Regularized(num_classes=len(DISEASES), dropout_rate=0.5)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        models.append(model)
        
        auc = checkpoint.get('best_auroc', 'N/A')
        seed = checkpoint.get('random_seed', 'N/A')
        auc_str = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
        print(f"  Modelo {i} carregado (seed {seed}, AUC {auc_str})")
    
    print(f"\nTotal: {len(models)} modelos prontos")
    return models

def ensemble_predict(models, dataloader, device):
    """
    Faz predições com ensemble (média das predições).
    
    Args:
        models: Lista de modelos treinados
        dataloader: DataLoader com dados
        device: Dispositivo
    
    Returns:
        Tuple (predições ensemble, labels verdadeiros)
    """
    all_ensemble_preds = []
    all_labels = []
        
    with torch.no_grad():
        for imgs, labels, _ in tqdm(dataloader, desc="Predicting"):
            imgs = imgs.to(device)
            
            batch_preds = []
            for model in models:
                outputs = model(imgs)
                probs = torch.sigmoid(outputs)
                batch_preds.append(probs)
            
            ensemble_pred = torch.stack(batch_preds).mean(dim=0)
            
            all_ensemble_preds.append(ensemble_pred.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_ensemble_preds = np.vstack(all_ensemble_preds)
    all_labels = np.vstack(all_labels)
    
    return all_ensemble_preds, all_labels

def evaluate_ensemble(ensemble_preds, labels, diseases):
    """
    Calcula AUC-ROC para ensemble.
    
    Args:
        ensemble_preds: Predições do ensemble
        labels: Labels verdadeiros
        diseases: Lista de nomes das doenças
    
    Returns:
        Dicionário com AUCs por doença e média
    """
    num_classes = len(diseases)
    aurocs = {}
    
    print("\nAVALIAÇÃO DO ENSEMBLE")
    print("="*70)
    
    for i, disease in enumerate(diseases):
        if len(np.unique(labels[:, i])) > 1:
            auc = roc_auc_score(labels[:, i], ensemble_preds[:, i])
            aurocs[disease] = auc
            print(f"{disease:30s} -> AUC: {auc:.4f}")
        else:
            aurocs[disease] = 0.5
            print(f"{disease:30s} -> AUC: 0.5000 (classe única)")
    
    mean_auc = np.mean(list(aurocs.values()))
    aurocs['mean'] = mean_auc
    
    print("="*70)
    print(f"AUC-ROC MÉDIO DO ENSEMBLE: {mean_auc:.4f}")
    print("="*70)
    
    return aurocs

def compare_individual_vs_ensemble(model_results, ensemble_auc, output_path):
    """
    Gera gráfico comparando modelos individuais vs ensemble.
    
    Args:
        model_results: Lista de resultados dos modelos
        ensemble_auc: AUC do ensemble
        output_path: Caminho para salvar gráfico
    """
    individual_aucs = [r['best_auroc'] for r in model_results]
    seeds = [r['random_seed'] for r in model_results]
    
    x_labels = [f"Modelo\n(seed {s})" for s in seeds] + ["ENSEMBLE"]
    aucs = individual_aucs + [ensemble_auc]
    colors = ['#3498db'] * len(individual_aucs) + ['#e74c3c']
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(aucs)), aucs, color=colors, alpha=0.8, edgecolor='black')
    
    mean_individual = np.mean(individual_aucs)
    plt.axhline(y=mean_individual, color='gray', linestyle='--', 
                label=f'Média Individual: {mean_individual:.4f}')
    
    for i, (bar, auc) in enumerate(zip(bars, aucs)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{auc:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('Modelos', fontsize=12)
    plt.ylabel('AUC-ROC', fontsize=12)
    plt.title('Comparação: Modelos Individuais vs Ensemble', fontsize=14, fontweight='bold')
    plt.xticks(range(len(aucs)), x_labels, fontsize=10)
    plt.ylim(min(aucs) - 0.02, max(aucs) + 0.03)
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    print(f"\nGráfico comparativo salvo em: {output_path}")
    plt.close()

def main():
    """Pipeline completo de ensemble training e avaliação."""
    
    print("\n" + "="*70)
    print("ENSEMBLE TRAINING PIPELINE")
    print("="*70 + "\n")
    
    train_new = input("Treinar novos modelos? (s/n): ").lower().strip() == 's'
    
    if train_new:
        model_results = train_ensemble_models(
            seeds=ENSEMBLE_SEEDS,
            base_dir=ENSEMBLE_DIR,
            debug=DEBUG_MODE
        )
        
        if not model_results:
            return
    else:
        print("\nUsando modelos existentes.")
        model_results = []
        for seed in ENSEMBLE_SEEDS:
            model_dir = os.path.join(ENSEMBLE_DIR, f"model_seed_{seed}")
            model_path = os.path.join(model_dir, "best_model.pt")
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                model_results.append({
                    'best_auroc': checkpoint.get('best_auroc', 0.0),
                    'random_seed': checkpoint.get('random_seed', seed),
                    'model_path': model_path
                })
    
    print("\n" + "="*70)
    print("AVALIANDO ENSEMBLE NO CONJUNTO DE VALIDAÇÃO")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    
    train_df_full = pd.read_csv(TRAIN_DF_PATH)
    if 'path' in train_df_full.columns:
        train_df_full.rename(columns={'path': 'Path'}, inplace=True)
    
    stratify_col = (train_df_full['Cardiomegaly'] == 1.0).astype(int)
    _, val_df = train_test_split(
        train_df_full.sample(n=100000, random_state=85),
        test_size=0.15,
        random_state=85,
        stratify=stratify_col.sample(n=100000, random_state=85)
    )
    
    _, val_transform = get_transforms()
    val_dataset = CheXpertDataset(val_df, PATH_IMAGE, val_transform, policy='u-ones')
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    model_paths = [r['model_path'] for r in model_results]
    models = load_ensemble_models(model_paths, device)
    
    if len(models) == 0:
        return
    
    ensemble_preds, labels = ensemble_predict(models, val_loader, device)
    
    aurocs = evaluate_ensemble(ensemble_preds, labels, DISEASES)
    
    output_comparison = os.path.join(ENSEMBLE_DIR, "comparison_individual_vs_ensemble.png")
    compare_individual_vs_ensemble(model_results, aurocs['mean'], output_comparison)
    
    results_summary = os.path.join(ENSEMBLE_DIR, "ensemble_results.csv")
    df_results = pd.DataFrame([aurocs])
    df_results.to_csv(results_summary, index=False)
    print(f"\nResultados salvos em: {results_summary}")
    
    print("\n" + "="*70)
    print("PIPELINE DE ENSEMBLE FINALIZADO")
    print("="*70)
    print(f"\nRESUMO:")
    print(f"  Modelos treinados: {len(models)}")
    print(f"  AUC médio individual: {np.mean([r['best_auroc'] for r in model_results]):.4f}")
    print(f"  AUC do ensemble: {aurocs['mean']:.4f}")
    ganho = (aurocs['mean'] - np.mean([r['best_auroc'] for r in model_results]))*100
    print(f"  Ganho: +{ganho:.2f}%")
    print(f"\nArquivos gerados:")
    print(f"  {output_comparison}")
    print(f"  {results_summary}")
    print(f"  Modelos em: {ENSEMBLE_DIR}/model_seed_*/\n")

if __name__ == "__main__":
    main()
