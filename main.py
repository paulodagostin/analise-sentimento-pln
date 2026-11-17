import pandas as pd

dados = pd.read_csv("data/avaliacoes.csv")

print(dados.head())

print("\nTotal de avaliações:")
print(len(dados))

print("\nQuantidade por sentimento:")
print(dados["sentimento"].value_counts())

print("\nTamanho médio das avaliações (em palavras):")
media = dados["texto"].str.split().str.len().mean()
print(round(media, 2))

import re

def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", " ", texto)
    texto = re.sub(r"\d+", " ", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()

dados["texto_limpo"] = dados["texto"].apply(limpar_texto)

print("\nExemplos de textos limpos:")
print(dados[["texto", "texto_limpo"]].head())

stopwords = {
    "a", "o", "as", "os", "um", "uma",
    "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas",
    "para", "por", "com", "que", "e"
}

def remover_stopwords(texto):
    palavras = texto.split()
    palavras_filtradas = [
        palavra for palavra in palavras
        if palavra not in stopwords
    ]
    return " ".join(palavras_filtradas)

dados["texto_sem_stopwords"] = dados["texto_limpo"].apply(remover_stopwords)

print("\nTextos sem stopwords:")
print(dados[["texto_limpo", "texto_sem_stopwords"]].head())

from sklearn.model_selection import train_test_split

X = dados["texto_limpo"]
y = dados["sentimento"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nQuantidade de avaliações para treino:", len(X_treino))
print("Quantidade de avaliações para teste:", len(X_teste))

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

bow = CountVectorizer()
tfidf = TfidfVectorizer()

X_treino_bow = bow.fit_transform(X_treino)
X_treino_tfidf = tfidf.fit_transform(X_treino)

print("\nBag of Words:")
print("Formato:", X_treino_bow.shape)

print("\nTF-IDF:")
print("Formato:", X_treino_tfidf.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

modelo_bow = LogisticRegression(max_iter=1000)

modelo_bow.fit(X_treino_bow, y_treino)

X_teste_bow = bow.transform(X_teste)

previsoes_bow = modelo_bow.predict(X_teste_bow)

acuracia_bow = accuracy_score(y_teste, previsoes_bow)

print("\nAcurácia do Bag of Words:")
print(f"{acuracia_bow:.2%}")

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

validacao = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

configuracoes = {
    "A - BoW com stopwords": CountVectorizer(),
    "B - TF-IDF com stopwords": TfidfVectorizer(),
    "C - BoW sem stopwords": CountVectorizer(stop_words=list(stopwords)),
    "D - TF-IDF sem stopwords": TfidfVectorizer(stop_words=list(stopwords))
}

print("\nComparação dos quatro modelos:")

for nome, vetorizador in configuracoes.items():

    pipeline = Pipeline([
        ("vetorizador", vetorizador),
        ("classificador", LogisticRegression(max_iter=1000))
    ])

    resultados_cv = cross_val_score(
        pipeline,
        X_treino,
        y_treino,
        cv=validacao,
        scoring="accuracy"
    )

    print(f"{nome}: {resultados_cv.mean():.2%}")

    modelo_final = Pipeline([
    ("vetorizador", CountVectorizer(stop_words=list(stopwords))),
    ("classificador", LogisticRegression(max_iter=1000))
])

modelo_final.fit(X_treino, y_treino)

previsoes_finais = modelo_final.predict(X_teste)

acuracia_final = accuracy_score(y_teste, previsoes_finais)

print("\nMelhor configuração: C - BoW sem stopwords")
print(f"Acurácia final: {acuracia_final:.2%}")

from sklearn.metrics import classification_report, confusion_matrix

print("\nRelatório de classificação:")
print(classification_report(y_teste, previsoes_finais, digits=3))

print("\nMatriz de confusão:")
print(confusion_matrix(
    y_teste,
    previsoes_finais,
    labels=["negativo", "positivo"]
))

erros = pd.DataFrame({
    "texto": X_teste,
    "sentimento_real": y_teste,
    "sentimento_previsto": previsoes_finais
})

erros = erros[
    erros["sentimento_real"] != erros["sentimento_previsto"]
]

print("\nAvaliações classificadas incorretamente:")
print(erros.to_string(index=False))

import os

os.makedirs("resultados", exist_ok=True)

probabilidades = modelo_final.predict_proba(X_teste)
classes = list(modelo_final.classes_)
indice_previsto = [classes.index(p) for p in previsoes_finais]

confiancas = [
    probabilidades[i, indice]
    for i, indice in enumerate(indice_previsto)
]

tabela_erros = pd.DataFrame({
    "texto": X_teste.to_numpy(),
    "real": y_teste.to_numpy(),
    "previsto": previsoes_finais,
    "probabilidade": confiancas
})

tabela_erros = tabela_erros[
    tabela_erros["real"] != tabela_erros["previsto"]
]

tabela_erros.to_csv(
    "resultados/erros.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArquivo de erros salvo com sucesso!")

import numpy as np

vetorizador_final = modelo_final.named_steps["vetorizador"]
classificador_final = modelo_final.named_steps["classificador"]

palavras = vetorizador_final.get_feature_names_out()
pesos = classificador_final.coef_[0]

indices_positivos = np.argsort(pesos)[-10:][::-1]
indices_negativos = np.argsort(pesos)[:10]

print("\n10 palavras mais associadas ao sentimento positivo:")

for indice in indices_positivos:
    print(palavras[indice], round(pesos[indice], 4))

print("\n10 palavras mais associadas ao sentimento negativo:")

for indice in indices_negativos:
    print(palavras[indice], round(pesos[indice], 4))

    top_positivas = pd.DataFrame({
    "palavra": palavras[indices_positivos],
    "peso": pesos[indices_positivos],
    "sentimento": "positivo"
})

top_negativas = pd.DataFrame({
    "palavra": palavras[indices_negativos],
    "peso": pesos[indices_negativos],
    "sentimento": "negativo"
})

top_palavras = pd.concat(
    [top_positivas, top_negativas],
    ignore_index=True
)

top_palavras.to_csv(
    "resultados/top_palavras.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArquivo top_palavras.csv salvo com sucesso!")