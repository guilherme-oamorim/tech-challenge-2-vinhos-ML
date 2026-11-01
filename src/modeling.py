"""Treina e avalia os modelos e salva gráficos e métricas em results/.

Uso (a partir da raiz do repositório):
    python src/modeling.py
"""

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocessing import RAIZ, carregar_dados, criar_alvo, dividir_e_padronizar, separar_x_y

PASTA_RESULTADOS = RAIZ / 'results'
COR_ALTA = '#7b1e3a'
COR_BAIXA = '#9e9e9e'


def treinar_modelos(X_train, y_train):
    """Treina os dois melhores modelos da comparação: SVM e Random Forest."""
    modelos = {
        'SVM': SVC(class_weight='balanced', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=300, random_state=42),
    }
    for modelo in modelos.values():
        modelo.fit(X_train, y_train)
    return modelos


def comparar_modelos(X, y, X_train, X_test, y_train, y_test):
    """Compara os seis modelos no teste e em validação cruzada estratificada (5 dobras)."""
    modelos = {
        'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=300, random_state=42),
        'Regressão Logística': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'SVM': SVC(class_weight='balanced', probability=True, random_state=42),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(),
        'Árvore de Decisão': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    }
    validacao = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    linhas = []
    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        proba = modelo.predict_proba(X_test)[:, 1]
        cv = cross_validate(make_pipeline(StandardScaler(), modelo), X, y, cv=validacao, scoring=['f1', 'roc_auc'])
        linhas.append({
            'modelo': nome,
            'f1_teste': f1_score(y_test, pred),
            'roc_auc_teste': roc_auc_score(y_test, proba),
            'f1_val_cruzada': cv['test_f1'].mean(),
            'roc_auc_val_cruzada': cv['test_roc_auc'].mean(),
        })
    return pd.DataFrame(linhas).set_index('modelo').round(3).sort_values('roc_auc_val_cruzada', ascending=False)


def salvar_comparacao(comparacao):
    comparacao.to_csv(PASTA_RESULTADOS / 'comparacao_modelos.csv')

    ordem = comparacao.sort_values('roc_auc_val_cruzada').index
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, (metrica, titulo) in zip(axes, [('f1', 'F1'), ('roc_auc', 'ROC-AUC')]):
        dados = comparacao.loc[ordem, [f'{metrica}_teste', f'{metrica}_val_cruzada']]
        dados.columns = [f'{titulo} (teste)', f'{titulo} (val. cruzada)']
        dados.plot(kind='barh', color=['#c06c84', COR_ALTA], ax=ax)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.set_title(titulo)
        ax.set_ylabel('')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9, frameon=False)
    fig.suptitle('Comparação entre os modelos', fontsize=14)
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / 'comparacao_modelos.png', dpi=150)
    plt.close(fig)


def avaliar_modelos(modelos, X_test, y_test):
    """Calcula as métricas da classe 'Alta qualidade' para cada modelo."""
    linhas = []
    for nome, modelo in modelos.items():
        pred = modelo.predict(X_test)
        proba = modelo.predict_proba(X_test)[:, 1]
        linhas.append({
            'modelo': nome,
            'precision': precision_score(y_test, pred),
            'recall': recall_score(y_test, pred),
            'f1': f1_score(y_test, pred),
            'acuracia': accuracy_score(y_test, pred),
            'roc_auc': roc_auc_score(y_test, proba),
        })
    return pd.DataFrame(linhas).set_index('modelo').round(3)


def salvar_balanceamento(df):
    contagem = df['alta_qualidade'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['Baixa/Média (0)', 'Alta (1)'], contagem.values, color=[COR_BAIXA, COR_ALTA])
    for i, v in enumerate(contagem.values):
        ax.text(i, v + 10, f'{v} ({v / contagem.sum():.1%})', ha='center')
    ax.set_ylim(0, contagem.max() * 1.12)
    ax.set_title('Balanceamento da variável alvo')
    ax.set_ylabel('Quantidade de vinhos')
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / 'balanceamento_classes.png', dpi=150)
    plt.close(fig)


def salvar_correlacao(df):
    corr = df.drop(columns=['Id', 'alta_qualidade']).corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title('Matriz de correlação')
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / 'matriz_correlacao.png', dpi=150)
    plt.close(fig)


def salvar_matrizes_confusao(modelos, X_test, y_test):
    fig, axes = plt.subplots(1, len(modelos), figsize=(10, 4))
    for ax, (nome, modelo) in zip(axes, modelos.items()):
        cm = confusion_matrix(y_test, modelo.predict(X_test))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False,
                    xticklabels=['Baixa/Média', 'Alta'], yticklabels=['Baixa/Média', 'Alta'], ax=ax)
        ax.set_title(nome)
        ax.set_xlabel('Predito')
        ax.set_ylabel('Real')
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / 'matrizes_confusao.png', dpi=150)
    plt.close(fig)


def salvar_curvas_roc(modelos, X_test, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))
    for (nome, modelo), cor in zip(modelos.items(), [COR_BAIXA, COR_ALTA]):
        proba = modelo.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, color=cor, label=f'{nome} (AUC = {roc_auc_score(y_test, proba):.3f})')
    ax.plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=0.8)
    ax.set_xlabel('Taxa de falsos positivos')
    ax.set_ylabel('Taxa de verdadeiros positivos')
    ax.set_title('Curva ROC')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / 'curvas_roc.png', dpi=150)
    plt.close(fig)


def salvar_importancias(modelo_rf, nomes_variaveis):
    importancias = pd.Series(modelo_rf.feature_importances_, index=nomes_variaveis).sort_values()
    importancias.sort_values(ascending=False).round(3).to_csv(
        PASTA_RESULTADOS / 'importancia_variaveis.csv', header=['importancia']
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importancias.index, importancias.values, color=COR_ALTA)
    ax.set_title('Importância das variáveis (Random Forest)')
    ax.set_xlabel('Importância')
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / 'importancia_variaveis.png', dpi=150)
    plt.close(fig)


def main():
    PASTA_RESULTADOS.mkdir(exist_ok=True)

    df = criar_alvo(carregar_dados())
    X, y = separar_x_y(df)
    X_train, X_test, y_train, y_test, _ = dividir_e_padronizar(X, y)

    modelos = treinar_modelos(X_train, y_train)

    comparacao = comparar_modelos(X, y, X_train, X_test, y_train, y_test)
    salvar_comparacao(comparacao)
    print(comparacao, end='\n\n')

    metricas = avaliar_modelos(modelos, X_test, y_test)
    metricas.to_csv(PASTA_RESULTADOS / 'metricas_modelos.csv')
    print(metricas)

    salvar_balanceamento(df)
    salvar_correlacao(df)
    salvar_matrizes_confusao(modelos, X_test, y_test)
    salvar_curvas_roc(modelos, X_test, y_test)
    salvar_importancias(modelos['Random Forest'], X.columns)

    print(f'\nResultados salvos em {PASTA_RESULTADOS}')


if __name__ == '__main__':
    main()
