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