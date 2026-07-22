"""
Orquestracao do treino do ensemble: treina os 5 modelos com seeds diferentes.

Esta versao NAO avalia o ensemble — a avaliacao vive em scripts dedicados
(analyze_ensemble.py, analyze_fairness.py, quantize_model.py), todos rodando
sobre o conjunto de teste comum. Aqui so treina.

Pre-requisito: rodar prepare_split.py UMA vez antes, para gerar o
common_test_set.csv que train_finetuning.py exclui do pool de treino.

Ordem geral do pipeline:
    1. python prepare_split.py          (gera o teste comum, uma vez)
    2. python train_ensemble.py         (treina os 5 modelos)   <- este script
    3. python analyze_ensemble.py       (AUC-14, AUC-5, individual vs ensemble)
    4. ajustar BEST_SEED em quantize_model.py conforme o passo 3
    5. python quantize_model.py         (quantizacao do melhor modelo)
    6. python analyze_fairness.py       (equidade por sexo e idade)
"""

import os
import numpy as np

from train_finetuning import train_model, COMMON_TEST_PATH

ENSEMBLE_SEEDS = [85, 42, 123, 777, 999]
ENSEMBLE_DIR = os.path.join('..', 'results_ensemble')
DEBUG_MODE = False


def train_ensemble_models(seeds, base_dir, debug=False):
    os.makedirs(base_dir, exist_ok=True)
    results = []

    print(f"\nTREINANDO ENSEMBLE COM {len(seeds)} MODELOS")
    print(f"Seeds: {seeds}")
    print(f"Diretório: {base_dir}\n")

    for i, seed in enumerate(seeds, 1):
        print(f"\n{'=' * 70}")
        print(f"MODELO {i}/{len(seeds)} - SEED {seed}")
        print(f"{'=' * 70}\n")

        model_dir = os.path.join(base_dir, f"model_seed_{seed}")

        try:
            result = train_model(random_seed=seed, results_dir=model_dir, debug_mode=debug)
            results.append(result)
            print(f"\nModelo {i} finalizado: AUC (val interna) {result['best_auroc']:.4f}")
        except Exception as e:
            print(f"\nErro ao treinar modelo {i} (seed {seed}): {e}")
            continue

    print(f"\n{'=' * 70}")
    print("SUMÁRIO DO TREINO")
    print(f"{'=' * 70}")
    for i, res in enumerate(results, 1):
        print(f"Modelo {i} (seed {res['random_seed']}): "
              f"AUC val interna {res['best_auroc']:.4f}")

    if results:
        aucs = [r['best_auroc'] for r in results]
        print(f"\nAUC de validacao interna (nao comparavel entre modelos "
              f"— cada um usa seu proprio split; a comparacao justa e' feita "
              f"depois por analyze_ensemble.py no teste comum):")
        print(f"  Média:  {np.mean(aucs):.4f}")
        print(f"  Desvio: {np.std(aucs):.4f}")

    return results


def main():
    if not os.path.exists(COMMON_TEST_PATH):
        print(f"ERRO: conjunto de teste comum nao encontrado em {COMMON_TEST_PATH}.")
        print("Rode 'python prepare_split.py' antes de treinar.")
        return

    print("\n" + "=" * 70)
    print("PIPELINE DE TREINO DO ENSEMBLE")
    print("=" * 70)

    train_ensemble_models(seeds=ENSEMBLE_SEEDS, base_dir=ENSEMBLE_DIR, debug=DEBUG_MODE)

    print("\n" + "=" * 70)
    print("TREINO CONCLUÍDO")
    print("=" * 70)
    print("Proximo passo: python analyze_ensemble.py")


if __name__ == "__main__":
    main()