"""Quantização pós-treino (float32 -> int8) do melhor modelo do ensemble.

Correcoes desta versao:
- Avalia no conjunto de teste comum (common_test_set.csv), e nao mais num
  "conjunto de validacao" reconstruido com random_state=85. Antes, o modelo da
  seed 42 era avaliado no split da seed 85, o que causava vazamento e inflava
  o AUC fp32 (0,7702 observado vs 0,7602 real do log de treino).
- BEST_SEED deve ser confirmado apos o retreino: o melhor modelo individual
  pode nao ser mais a seed 42. Rode analyze_ensemble.py antes e ajuste aqui.
"""

import os
import copy
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

try:
    from torch.ao.quantization import quantize_fx, get_default_qconfig
    from torch.ao.quantization.qconfig_mapping import QConfigMapping
except ImportError:
    from torch.quantization import quantize_fx, get_default_qconfig
    from torch.quantization.qconfig_mapping import QConfigMapping

from train_finetuning import (
    DenseNet121_Regularized,
    CheXpertDataset,
    get_transforms,
    DISEASES,
    TRAIN_DF_PATH,
    PATH_IMAGE,
)

RESULTS_DIR = os.path.join('..', 'results_ensemble')
OUTPUT_DIR = os.path.join('..', 'results_quantization')
COMMON_TEST_PATH = os.path.join(RESULTS_DIR, 'common_test_set.csv')

BEST_SEED = 777

BATCH_SIZE_EVAL = 32
WORKERS = 6
CALIBRATION_BATCHES = 20
BENCHMARK_BATCHES = 15

COMPETITION_PATHOLOGIES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']


def build_test_loader(batch_size=BATCH_SIZE_EVAL):
    """Carrega o conjunto de teste comum, o mesmo usado por analyze_ensemble."""
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

    _, val_transform = get_transforms()
    dataset = CheXpertDataset(test_df, PATH_IMAGE, val_transform, policy='u-ones')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=WORKERS, pin_memory=True)
    return loader, len(test_df)


def load_fp32_model():
    model_path = os.path.join(RESULTS_DIR, f"model_seed_{BEST_SEED}", "best_model.pt")
    checkpoint = torch.load(model_path, map_location='cpu')

    model = DenseNet121_Regularized(num_classes=len(DISEASES), dropout_rate=0.5)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def calibrate(model_prepared, loader, n_batches=CALIBRATION_BATCHES):
    model_prepared.eval()
    with torch.no_grad():
        for i, (imgs, _, _) in enumerate(loader):
            if i >= n_batches:
                break
            model_prepared(imgs)


def quantize_static_fx(model_fp32, calibration_loader):
    model_fp32 = model_fp32.to('cpu').eval()
    qconfig_mapping = QConfigMapping().set_global(get_default_qconfig('fbgemm'))
    example_inputs = (next(iter(calibration_loader))[0].to('cpu'),)

    model_prepared = quantize_fx.prepare_fx(model_fp32, qconfig_mapping, example_inputs)
    calibrate(model_prepared, calibration_loader)
    return quantize_fx.convert_fx(model_prepared)


def quantize_dynamic_fallback(model_fp32):
    return torch.quantization.quantize_dynamic(
        model_fp32.to('cpu').eval(), {torch.nn.Linear}, dtype=torch.qint8
    )


def evaluate(model, loader, device='cpu', num_classes=14):
    model = model.to(device)
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_labels.append(labels.numpy())
            all_probs.append(probs)

    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)

    aucs = []
    for i in range(num_classes):
        if len(np.unique(all_labels[:, i])) > 1:
            aucs.append(roc_auc_score(all_labels[:, i], all_probs[:, i]))
        else:
            aucs.append(np.nan)
    return aucs


def model_size_mb(model, path="_temp_size_check.pt"):
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / 1e6
    os.remove(path)
    return size_mb


def benchmark_inference(model, loader, device='cpu', n_batches=BENCHMARK_BATCHES):
    model = model.to(device)
    model.eval()

    batches = []
    for i, (imgs, _, _) in enumerate(loader):
        if i >= n_batches:
            break
        batches.append(imgs.to(device))

    with torch.no_grad():
        for imgs in batches[:2]:
            model(imgs)

        start = time.time()
        for imgs in batches:
            model(imgs)
        elapsed = time.time() - start

    return elapsed / len(batches)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == 'cuda' else ""))
    print(f"Modelo quantizado: seed {BEST_SEED}")

    model_fp32 = load_fp32_model()
    test_loader, n_test = build_test_loader()
    print(f"Teste comum: {n_test:,} imagens")

    try:
        model_int8 = quantize_static_fx(copy.deepcopy(model_fp32), test_loader)
        method = "estatica_fx"
    except Exception as e:
        print(f"Quantização estática falhou ({e}); usando fallback dinâmico.")
        model_int8 = quantize_dynamic_fallback(copy.deepcopy(model_fp32))
        method = "dinamica_fallback"

    aucs_fp32 = evaluate(model_fp32, test_loader, device=device)
    aucs_int8 = evaluate(model_int8, test_loader, device='cpu')

    size_fp32 = model_size_mb(model_fp32.to('cpu'))
    size_int8 = model_size_mb(model_int8)

    time_fp32 = benchmark_inference(model_fp32, test_loader, device='cpu')
    time_int8 = benchmark_inference(model_int8, test_loader, device='cpu')

    idx_5 = [DISEASES.index(p) for p in COMPETITION_PATHOLOGIES]
    auc14_fp32, auc14_int8 = np.nanmean(aucs_fp32), np.nanmean(aucs_int8)
    auc5_fp32 = np.nanmean([aucs_fp32[i] for i in idx_5])
    auc5_int8 = np.nanmean([aucs_int8[i] for i in idx_5])

    print(f"\nMétodo: {method}")
    print(f"Tamanho: fp32 {size_fp32:.2f} MB | int8 {size_int8:.2f} MB "
          f"({size_fp32 / size_int8:.2f}x menor, -{100 * (1 - size_int8 / size_fp32):.1f}%)")
    print(f"AUC 14 patologias: fp32 {auc14_fp32:.4f} | int8 {auc14_int8:.4f} ({auc14_int8 - auc14_fp32:+.4f})")
    print(f"AUC 5 patologias: fp32 {auc5_fp32:.4f} | int8 {auc5_int8:.4f} ({auc5_int8 - auc5_fp32:+.4f})")
    print(f"Tempo/batch (CPU): fp32 {time_fp32 * 1000:.1f} ms | int8 {time_int8 * 1000:.1f} ms "
          f"({time_fp32 / time_int8:.2f}x mais rápido)")

    pd.DataFrame({
        'Disease': DISEASES,
        'AUC_fp32': aucs_fp32,
        'AUC_int8': aucs_int8,
        'Delta': [b - a for a, b in zip(aucs_fp32, aucs_int8)],
    }).to_csv(os.path.join(OUTPUT_DIR, 'quantization_per_disease.csv'), index=False)

    pd.DataFrame([{
        'Seed': BEST_SEED,
        'Method': method,
        'Size_MB_fp32': size_fp32,
        'Size_MB_int8': size_int8,
        'Size_reduction_x': size_fp32 / size_int8,
        'AUC_14_fp32': auc14_fp32,
        'AUC_14_int8': auc14_int8,
        'AUC_5_fp32': auc5_fp32,
        'AUC_5_int8': auc5_int8,
        'Time_ms_fp32_cpu': time_fp32 * 1000,
        'Time_ms_int8_cpu': time_int8 * 1000,
    }]).to_csv(os.path.join(OUTPUT_DIR, 'quantization_summary.csv'), index=False)

    torch.save(model_int8.state_dict(), os.path.join(OUTPUT_DIR, f'model_seed_{BEST_SEED}_int8.pt'))
    print(f"Resultados salvos em: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()