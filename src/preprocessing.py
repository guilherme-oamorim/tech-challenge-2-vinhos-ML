"""Funções de pré-processamento da base de vinhos (mesmas etapas do notebook)."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
CAMINHO_DADOS = RAIZ / 'data' / 'WineQT.csv'

NOTA_CORTE = 7


def carregar_dados(caminho=CAMINHO_DADOS):
    """Lê o CSV de vinhos."""
    return pd.read_csv(caminho)


def criar_alvo(df, nota_corte=NOTA_CORTE):
    """Cria a coluna binária `alta_qualidade`: 1 se quality >= nota_corte, senão 0."""
    df = df.copy()
    df['alta_qualidade'] = (df['quality'] >= nota_corte).astype(int)
    return df


def separar_x_y(df):
    """Separa as 11 variáveis físico-químicas (X) do alvo (y).

    `quality` fica de fora do X porque o alvo é derivado dela (vazamento de dados).
    """
    X = df.drop(columns=['Id', 'quality', 'alta_qualidade'])
    y = df['alta_qualidade']
    return X, y


def dividir_e_padronizar(X, y, test_size=0.2, random_state=42):
    """Divide em treino/teste estratificado e padroniza com o scaler ajustado só no treino."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
