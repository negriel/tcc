"""
Prepara o conjunto de teste comum, reservado UMA vez antes de qualquer treino.

Este é o passo que corrige o vazamento de dados: em vez de cada script
reconstruir seu proprio "conjunto de validacao" reamostrando os dados na hora
(com seeds que muitas vezes nao batem com a seed que treinou o modelo), aqui
separa-se um unico conjunto de teste, salvo em disco, que todos os scripts
seguintes leem. Nenhum dos 5 modelos treina nas imagens que caem aqui.

Rodar UMA vez, antes de train_ensemble.py:
    python prepare_split.py

Gera: ../results_ensemble/common_test_set.csv  (coluna 'Path' das imagens de teste)
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_DF_PATH = r"E:\GabrielRibeiro\chexpert_project\data\CheXpert-v1.0-small\train.csv"
OUTPUT_DIR = os.path.join('..', 'results_ensemble')
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'common_test_set.csv')

# Seed do split de teste. Deliberadamente diferente das 5 seeds dos modelos
# (85, 42, 123, 777, 999) para deixar explicito que o conjunto de teste nao
# esta amarrado a nenhum modelo especifico.
TEST_SPLIT_SEED = 2024
TEST_SIZE = 20000


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(TRAIN_DF_PATH)
    if 'path' in df.columns:
        df.rename(columns={'path': 'Path'}, inplace=True)

    # Estratifica pela presenca de Cardiomegaly, mesmo criterio ja usado no
    # split interno de treino/validacao, para manter o conjunto de teste
    # representativo em termos de prevalencia.
    stratify_col = (df['Cardiomegaly'] == 1.0).astype(int)

    _, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=TEST_SPLIT_SEED,
        stratify=stratify_col,
    )

    test_df = test_df.reset_index(drop=True)
    test_df[['Path']].to_csv(OUTPUT_PATH, index=False)

    print(f"Conjunto de teste comum reservado: {len(test_df):,} imagens")
    print(f"Salvo em: {OUTPUT_PATH}")
    print(f"Prevalencia de Cardiomegaly no teste: "
          f"{(test_df['Cardiomegaly'] == 1.0).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()