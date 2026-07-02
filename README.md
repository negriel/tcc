# CheXpert Multi-Label Classification - Ensemble Learning

## Descrição

Classificação multi-rótulo de radiografias de tórax usando DenseNet-121 com ensemble de 5 modelos.

Dataset: CheXpert-v1.0-small (Stanford ML Group)  
Doenças: 14 patologias torácicas  
Arquitetura: DenseNet-121 (ImageNet)

## Estrutura do Projeto

O projeto está organizado da seguinte forma:

- `data/` contém o dataset (não versionado)
- `src/` contém os códigos principais
  - <span class="truncated-code-wrapper" data-full-text="train_optimized.py" title="train_optimized.py"><code class="truncated-code">train_optimized…</code><span class="copy-code-inline-btn" data-copy-text="train_optimized.py"></span></span> implementa o baseline com transfer learning
  - <span class="truncated-code-wrapper" data-full-text="train_finetuning.py" title="train_finetuning.py"><code class="truncated-code">train_finetunin…</code><span class="copy-code-inline-btn" data-copy-text="train_finetuning.py"></span></span> implementa fine-tuning progressivo
  - <span class="truncated-code-wrapper" data-full-text="train_ensemble.py" title="train_ensemble.py"><code class="truncated-code">train_ensemble.…</code><span class="copy-code-inline-btn" data-copy-text="train_ensemble.py"></span></span> treina ensemble de 5 modelos
  - <span class="truncated-code-wrapper" data-full-text="analyse_ensemble.py" title="analyse_ensemble.py"><code class="truncated-code">analyse_ensembl…</code><span class="copy-code-inline-btn" data-copy-text="analyse_ensemble.py"></span></span> avalia o ensemble
  - <span class="truncated-code-wrapper" data-full-text="fairness_analysis.py" title="fairness_analysis.py"><code class="truncated-code">fairness_analys…</code><span class="copy-code-inline-btn" data-copy-text="fairness_analysis.py"></span></span> análise de equidade
- `results/` armazena resultados do baseline
- <span class="truncated-code-wrapper" data-full-text="results_ensemble/" title="results_ensemble/"><code class="truncated-code">results_ensembl…</code><span class="copy-code-inline-btn" data-copy-text="results_ensemble/"></span></span> armazena resultados do ensemble
- <span class="truncated-code-wrapper" data-full-text="results_fairness/" title="results_fairness/"><code class="truncated-code">results_fairnes…</code><span class="copy-code-inline-btn" data-copy-text="results_fairness/"></span></span> armazena análise de equidade

## Como Usar

### 1. Criar Ambiente

```bash
conda create -n chexpert python=3.10 -y
conda activate chexpert
```text


```bash
pip install -r requirements.txt
```text


Acesse https://stanfordmlgroup.github.io/competitions/chexpert/ e descompacte em <span class="truncated-code-wrapper" data-full-text="data/CheXpert-v1.0-small/" title="data/CheXpert-v1.0-small/"><code class="truncated-code">data/CheXpert-v…</code><span class="copy-code-inline-btn" data-copy-text="data/CheXpert-v1.0-small/"></span></span>


Navegue até a pasta src:

```bash
cd src
```text

Baseline:

```bash
python train_optimized.py
```text

Fine-tuning progressivo:

```bash
python train_finetuning.py
```text

Ensemble:

```bash
python train_ensemble.py
```text

### 5. Análises

Avaliar ensemble:

```bash
python analyse_ensemble.py
```text

Análise de equidade:

```bash
python fairness_analysis.py
```text


| Experimento | Estratégia | AUC-ROC |
|-------------|-----------|---------|
| Baseline Otimizado | Transfer learning | 0.7516 |
| Fine-tuning Progressivo | Descongelamento gradual | 0.7602 |
| Ensemble (5 modelos) | Agregação de previsões | 0.7643 |


- Sexo: Disparidade menor que 1%
- Idade: Variância menor que 0.001


- Python 3.10+
- PyTorch 2.0+
- DenseNet-121 (ImageNet)
- scikit-learn
- pandas
- matplotlib
- seaborn

Gabriel Ribeiro  
Trabalho de conclusão de curso do curso de Ciência da Computação - UFU

## Referências

- Irvin, J., et al. (2019). CheXpert: A Large Chest Radiograph Dataset. AAAI.
- Huang, G., et al. (2017). Densely Connected Convolutional Networks. CVPR.
