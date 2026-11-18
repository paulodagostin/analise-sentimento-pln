from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "resultados"
RESULTADOS.mkdir(exist_ok=True)
STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "para", "por", "com", "que", "e"
}


def limpar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^\w\s]", " ", texto)
    texto = re.sub(r"\d+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def criar_pipeline(vetorizador):
    return Pipeline([
        ("vetorizador", vetorizador),
        ("classificador", LogisticRegression(max_iter=1000, random_state=42)),
    ])

def main():
    dados = pd.read_csv(BASE / "data" / "avaliacoes.csv")
    dados["texto_limpo"] = dados["texto"].apply(limpar_texto)
    media_por_classe = dados.groupby("sentimento")["texto"].apply(
        lambda coluna: coluna.str.split().str.len().mean()
    )
    with (RESULTADOS / "exploracao.txt").open("w", encoding="utf-8") as arq:
        arq.write(f"Total de avaliações: {len(dados)}\n\n")
        arq.write("Distribuição dos sentimentos:\n")
        arq.write(dados["sentimento"].value_counts().to_string())
        arq.write("\n\nMédia de palavras por classe:\n")
        arq.write(media_por_classe.to_string())
        arq.write("\n\nTrês exemplos por classe:\n")
        for classe in ("positivo", "negativo"):
            arq.write(f"\n{classe.upper()}:\n")
            for exemplo in dados.loc[dados["sentimento"] == classe, "texto"].head(3):
                arq.write(f"- {exemplo}\n")

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        dados["texto_limpo"], dados["sentimento"],
        test_size=0.25, random_state=42, stratify=dados["sentimento"]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    configuracoes = [
        ("A", "mantidas", "BoW", CountVectorizer()),
        ("B", "mantidas", "TF-IDF", TfidfVectorizer()),
        ("C", "removidas", "BoW", CountVectorizer(stop_words=sorted(STOPWORDS))),
        ("D", "removidas", "TF-IDF", TfidfVectorizer(stop_words=sorted(STOPWORDS))),
    ]
    linhas = []
    for codigo, stopwords, nome_vetorizador, vetorizador in configuracoes:
        pipeline = criar_pipeline(vetorizador)
        metricas = cross_validate(
            pipeline, X_treino, y_treino, cv=cv,
            scoring={"acc": "accuracy", "f1": "f1_macro"},
        )
        pipeline.fit(X_treino, y_treino)
        linhas.append({
            "config": codigo,
            "stopwords": stopwords,
            "vetorizador": nome_vetorizador,
            "vocabulario": len(pipeline.named_steps["vetorizador"].get_feature_names_out()),
            "acc_cv_media": metricas["test_acc"].mean(),
            "acc_cv_desvio": metricas["test_acc"].std(),
            "f1_cv_media": metricas["test_f1"].mean(),
            "f1_cv_desvio": metricas["test_f1"].std(),
        })
    comparacao = pd.DataFrame(linhas)
    comparacao.to_csv(RESULTADOS / "comparacao.csv", index=False, encoding="utf-8-sig")
    # Critério: maior acurácia média; empate desempatado pelo F1 macro médio;
    # se persistir, menor vocabulário; depois, ordem alfabética da configuração.
    vencedor = comparacao.sort_values(
        ["acc_cv_media", "f1_cv_media", "vocabulario", "config"],
        ascending=[False, False, True, True], kind="stable"
    ).iloc[0]
    codigo = vencedor["config"]
    vetor_final = next(v for c, _, _, v in configuracoes if c == codigo)
    modelo_final = criar_pipeline(vetor_final)
    modelo_final.fit(X_treino, y_treino)
    previsoes = modelo_final.predict(X_teste)
    probabilidades = modelo_final.predict_proba(X_teste)
    classes = modelo_final.classes_.tolist()

    matriz = confusion_matrix(y_teste, previsoes, labels=["negativo", "positivo"])
    pd.DataFrame(
        matriz,
        index=["real_negativo", "real_positivo"],
        columns=["previsto_negativo", "previsto_positivo"],
    ).to_csv(RESULTADOS / "matriz_confusao.csv", encoding="utf-8-sig")

    relatorio = classification_report(y_teste, previsoes, digits=3)
    (RESULTADOS / "relatorio_classificacao.txt").write_text(
        f"Modelo escolhido: {codigo}\nAcurácia final: {accuracy_score(y_teste, previsoes):.4f}\n\n{relatorio}",
        encoding="utf-8",
    )

    confianca_prevista = [probabilidades[i, classes.index(c)] for i, c in enumerate(previsoes)]
    erros = pd.DataFrame({
        "texto": X_teste.to_numpy(),
        "real": y_teste.to_numpy(),
        "previsto": previsoes,
        "probabilidade": confianca_prevista,
    })
    erros.loc[erros["real"] != erros["previsto"]].to_csv(
        RESULTADOS / "erros.csv", index=False, encoding="utf-8-sig"
    )

    vet = modelo_final.named_steps["vetorizador"]
    palavras = vet.get_feature_names_out()
    pesos = modelo_final.named_steps["classificador"].coef_[0]
    # Para classificador binário, coef_[0] favorece classes_[1] quando positivo.
    if classes[1] != "positivo":
        pesos = -pesos
    negativos, positivos = np.argsort(pesos)[:10], np.argsort(pesos)[-10:][::-1]
    influentes = pd.concat([
        pd.DataFrame({"palavra": palavras[positivos], "peso": pesos[positivos], "sentimento": "positivo"}),
        pd.DataFrame({"palavra": palavras[negativos], "peso": pesos[negativos], "sentimento": "negativo"}),
    ], ignore_index=True)
    influentes.to_csv(RESULTADOS / "palavras_influentes.csv", index=False, encoding="utf-8-sig")

    casos = pd.read_csv(BASE / "data" / "casos_dificeis.csv")
    textos = casos["texto"].apply(limpar_texto)
    pred_casos = modelo_final.predict(textos)
    prob_positivo = modelo_final.predict_proba(textos)[:, classes.index("positivo")]
    pd.DataFrame({
        "texto": casos["texto"],
        "tipo": casos["tipo"],
        "esperado": casos["sentimento_esperado"],
        "previsto": pred_casos,
        "probabilidade_positivo": prob_positivo,
    }).to_csv(RESULTADOS / "casos_dificeis_predicoes.csv", index=False, encoding="utf-8-sig")

    print(f"Avaliações: {len(dados)} | treino: {len(X_treino)} | teste: {len(X_teste)}")
    print("\nComparação (CV):\n", comparacao.to_string(index=False))
    print(f"\nModelo vencedor: {codigo} | acurácia no teste: {accuracy_score(y_teste, previsoes):.2%}")
    print("\nMatriz de confusão:\n", matriz)
    print("\nArquivos criados em resultados/:")
    for arquivo in sorted(RESULTADOS.iterdir()):
        if arquivo.name in {"comparacao.csv", "exploracao.txt", "matriz_confusao.csv", "relatorio_classificacao.txt", "erros.csv", "palavras_influentes.csv", "casos_dificeis_predicoes.csv"}:
            print(" -", arquivo.name)


if __name__ == "__main__":
    main()