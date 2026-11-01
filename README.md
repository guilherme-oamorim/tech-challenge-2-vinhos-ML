# Classificação da Qualidade de Vinhos com Machine Learning

Tech Challenge — Fase 2 | Pós-Tech Data Analytics (FIAP)

## Contexto

A qualidade de um vinho costuma ser avaliada por especialistas em análises sensoriais, um processo subjetivo, demorado e dependente da experiência de quem avalia. Este projeto usa as medidas físico-químicas coletadas durante a produção para prever se um vinho será de alta qualidade, apoiando enólogos e produtores na tomada de decisão.

## Objetivo

Construir um modelo de classificação que, a partir de 11 características físico-químicas, preveja se um vinho é de **alta qualidade**:

| Classe | Critério |
|---|---|
| Alta qualidade (1) | nota (`quality`) ≥ 7 |
| Baixa/Média qualidade (0) | nota (`quality`) < 7 |

## Base de dados

[Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) (Kaggle): 1143 amostras de vinho tinto, com acidez fixa e volátil, ácido cítrico, açúcar residual, cloretos, dióxido de enxofre livre e total, densidade, pH, sulfatos, teor alcoólico e a nota dada por especialistas.

## Estrutura do repositório

```
├── data/              # Base de dados utilizada (WineQT.csv)
├── notebooks/         # Notebook com a análise e modelagem
├── src/               # Scripts auxiliares (pré-processamento ou modelagem)
├── results/           # Gráficos e métricas dos modelos
├── requirements.txt   # Bibliotecas utilizadas
└── README.md          # Descrição do projeto
```

## Como executar

```bash
pip install -r requirements.txt
```

**Análise completa:** abra `notebooks/Introduction.ipynb` no VS Code ou no Jupyter e execute todas as células. O notebook lê os dados de `../data/WineQT.csv`, então deve ser executado a partir da pasta `notebooks/`.

**Apenas o pipeline de modelagem:** a partir da raiz do repositório, execute

```bash
python src/modeling.py
```

O script reproduz o pré-processamento e o treino do notebook e salva em `results/`:

| Arquivo | Conteúdo |
|---|---|
| `comparacao_modelos.csv` / `.png` | F1 e ROC-AUC dos 6 modelos comparados, no teste e na validação cruzada |
| `metricas_modelos.csv` | Precision, recall, F1, acurácia e ROC-AUC dos dois melhores modelos (Random Forest e SVM) |
| `importancia_variaveis.csv` / `.png` | Importância das variáveis no Random Forest |
| `matrizes_confusao.png` | Matrizes de confusão dos dois modelos |
| `curvas_roc.png` | Curvas ROC dos dois modelos |
| `matriz_correlacao.png` | Correlação entre as variáveis e a nota |
| `balanceamento_classes.png` | Distribuição da variável alvo |

Os scripts em `src/` são:
- `preprocessing.py`: leitura dos dados, criação do alvo binário, separação X/y, divisão treino/teste e padronização;
- `modeling.py`: treino, avaliação e geração dos gráficos e métricas.

## Etapas

1. **Compreensão do problema**: definição do alvo binário `alta_qualidade`.
2. **Análise exploratória**: distribuições, assimetria, outliers, comparação entre classes e correlações.
3. **Pré-processamento**: separação treino/teste estratificada (80/20) e padronização com `StandardScaler` ajustado só no treino.
4. **Modelagem**: comparação de 6 modelos (Random Forest, SVM, Regressão Logística, Naive Bayes, KNN e Árvore de Decisão), no teste e em validação cruzada estratificada de 5 dobras. Usamos `class_weight='balanced'` nos modelos que oferecem essa opção, por causa do desbalanceamento (86,1% / 13,9%). Os dois melhores pela ROC-AUC seguem para as próximas etapas.
5. **Avaliação**: precision, recall, F1, ROC-AUC e matriz de confusão.
6. **Interpretação**: importância das variáveis e implicações para a produção.

## Resultados

Comparação dos modelos (classe Alta qualidade):

| Modelo | F1 (teste) | ROC-AUC (teste) | F1 (val. cruzada) | ROC-AUC (val. cruzada) |
|---|---|---|---|---|
| **Random Forest** | 0.597 | 0.913 | 0.676 | 0.935 |
| **SVM** | 0.523 | 0.870 | 0.535 | 0.891 |
| Regressão Logística | 0.489 | 0.850 | 0.510 | 0.873 |
| Naive Bayes | 0.514 | 0.810 | 0.544 | 0.851 |
| KNN | 0.475 | 0.834 | 0.476 | 0.843 |
| Árvore de Decisão | 0.562 | 0.746 | 0.525 | 0.727 |

Os dois melhores pela ROC-AUC na validação cruzada foram o **Random Forest** e o **SVM**.

Métricas no conjunto de teste dos dois modelos escolhidos:

| Métrica | Random Forest | SVM |
|---|---|---|
| Precision | 0.571 | 0.411 |
| Recall | 0.625 | 0.719 |
| F1-score | 0.597 | 0.523 |
| Acurácia | 0.882 | 0.817 |
| ROC-AUC | 0.913 | 0.870 |

O **Random Forest** é o modelo principal, por ter o melhor desempenho geral e menos erros no total. O **SVM** é a alternativa para quando a prioridade for não deixar passar vinhos bons, por ter o maior recall.

## Principais conclusões

- **Teor alcoólico** é a variável mais influente: vinhos com mais álcool tendem a receber notas maiores.
- **Sulfatos** contribuem positivamente para a qualidade, e o **dióxido de enxofre total** tende a ser menor nos vinhos bons.
- **Ácido cítrico** é mais alto nos vinhos bons, mas seu peso varia entre os modelos.
- **Acidez volátil** alta está associada a vinhos piores (defeito de sabor semelhante a vinagre).
- **Açúcar residual** e **pH** têm pouca influência.

São relações estatísticas, não causais. O modelo deve ser usado como apoio à decisão (por exemplo, para priorizar lotes para análise sensorial), e não como substituto da avaliação de especialistas.
