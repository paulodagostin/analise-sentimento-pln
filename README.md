# Análise de Sentimento com PLN

Projeto desenvolvido para a disciplina de Inteligência Artificial e Processamento de Linguagem Natural, utilizando Python para classificar avaliações de produtos em português como positivas ou negativas.

## Objetivo

O objetivo da atividade é comparar duas técnicas de representação de textos, Bag of Words (BoW) e TF-IDF, utilizando Regressão Logística para identificar sentimentos nas avaliações.

Também foram realizados testes com validação cruzada, análise dos erros e avaliação de frases mais difíceis, como ironias e negações.

## Tecnologias utilizadas

- Python 3.13
- Pandas
- NumPy
- Scikit-learn

## Como executar

Primeiro, instale as bibliotecas necessárias:

```bash
python -m pip install -r requirements.txt
```

Depois, execute o programa:

```bash
python main.py
```

Os resultados serão gerados automaticamente na pasta `resultados/`.

## Resultados

Foram utilizadas 260 avaliações, sendo 130 positivas e 130 negativas.

A configuração escolhida foi Bag of Words sem remoção das palavras de negação, com remoção das stopwords definidas no projeto.

O modelo apresentou acurácia de 85,64% na validação cruzada e 84,62% no conjunto de teste.

## Arquivos do projeto

- `main.py`: código principal do classificador.
- `data/`: arquivos CSV utilizados.
- `resultados/`: métricas, previsões e análises geradas.
- `RELATORIO.md`: respostas e análise dos resultados.
- `requirements.txt`: bibliotecas necessárias para executar o projeto.

## Autor

Paulo Ricardo Dagostin Rosso