# RELATORIO — Análise de Sentimento com PLN

**Nome: Paulo Ricardo Dagostin Rosso** 
**Repositório:https://github.com/paulodagostin/analise-sentimento-pln** 
**Data: 17/11/2026** 

> Responda cada pergunta com **4 a 12 linhas**, com suas palavras e usando números dos arquivos de `resultados/`. Apague as instruções em itálico ao entregar.

## Tabela-resumo (preencha com os seus resultados)

| Config | Stopwords | Vetorizador | Vocabulário | Acurácia CV (média ± desvio) | F1 CV (média ± desvio) |
|---|---|---|---|---|---|
| A | mantidas | BoW | 420 | 0,8564 ± 0,0926 | 0,8563 ± 0,0926 |
| B | mantidas | TF-IDF | 420 | 0,8462 ± 0,0843 | 0,8459 ± 0,0844 |
| C | removidas | BoW | 407 | 0,8564 ± 0,0804 | 0,8563 ± 0,0804 |
| D | removidas | TF-IDF | 407 | 0,8462 ± 0,0725 | 0,8458 ± 0,0727 |

**Configuração escolhida:** C (BoW sem stopwords) **Acurácia no teste:** 0,8462 (84,62%) **F1 no teste:** 0,846 (macro)

---

## 1. Pré-processamento
Comecei convertendo os textos para letras minúsculas e eliminando as diferenças entre palavras de "Ótimo" e "ótimo", também removi pontuações, números e espaços repetidos. foi criada uma lista com 21 stopwords em português e o vocabulário passou de 420 para 407 após a remoção, preservei a palavra "não" porque pode inverter o sentimento de uma frase e tambem "nunca" e "nem", pelo mesmo motivo.

## 2. BoW vs. TF-IDF
 O BoW conta quantas vezes uma palavra aparece no texto, já o TF-IDF considera a frequência e a importância dela em relação aos outros textos. Comparando os resultados, o BoW teve uma acurácia de 85,64%, enquanto o TF-IDF ficou com 84,62%. A diferença foi pequena, de apenas 1,02 ponto percentual, principalmente considerando os desvios padrões.

Escolhi o BoW sem stopwords porque apresentou uma das maiores acurácias médiasve teve um desvio padrão menor que o BoW com stopwords. Isso mostrou uma variação menor entre os testes.

## 3. Avaliação no teste
Na avaliação final, o modelo teve uma acurácia de 84,62%, acertando 55 das 65 avaliações. Na matriz de confusão, foram 26 negativos e 29 positivos classificados corretamente, enquanto 7 negativos foram considerados positivos e 3 positivos foram considerados negativos. A classe negativa teve precisão de 89,7%, recall de 78,8% e F1 de 83,9%. Já a positiva teve precisão de 80,6%, recall de 90,6% e F1 de 85,3%.

Pensando em uma loja, imagino que o erro mais grave seria classificar uma reclamação como elogio, porque a empresa poderia deixar de perceber um problema de um cliente. A acurácia é menos enganosa nesse caso porque temos 130 avaliações positivas e 130 negativas, diferente de um cenário de fraude, em que normalmente há muito mais casos legítimos do que fraudulentos.

## 4. Validação cruzada vs. divisão única
A validação cruzada é mais confiável porque divide os dados de treinamento em cinco partes e testa o modelo várias vezes, em vez de depender de apenas uma divisão. Assim conseguimos comparar os resultados e perceber melhor as variações. Não podemos escolher o melhor modelo olhando o teste final, porque isso poderia influenciar a escolha e deixar o resultado menos confiável. O teste deve servir apenas para avaliar o modelo escolhido.

O vazamento de dados acontece quando o modelo recebe informações que não deveria conhecer durante o treinamento. Para evitar isso, utilizei o Pipeline, que faz o vetorizador aprender apenas com os dados de treino de cada divisão, sem utilizar os dados de validação ou teste.

## 5. Análise de erros e palavras influentes
Analisando os erros, percebi que o modelo teve dificuldades em algumas avaliações. Por exemplo, classificou "cheguei animado mas me arrependi tudo estava sujo" como positivo, mesmo tendo palavras negativas. Também errou "panela de péssima qualidade a comida gruda e queima", considerando como positivo, e "a loja é confiável e o suporte respondeu rapidinho", classificando como negativo.

Nas palavras mais influentes, algumas como "bastante", "tudo" e "mais" não representam necessariamente um sentimento sozinhas. Mesmo assim, receberam pesos importantes no modelo. Isso mostra que ele aprendeu associações entre as palavras e os sentimentos presentes nos dados, mas nem sempre consegue entender o contexto completo de uma frase.

## 6. Casos difíceis e limitações
Nos casos difíceis, percebi que o modelo teve problemas principalmente com frases que dependem de contexto. Na negação, errou frases como "não tenho nada a reclamar, tudo perfeito". Na ironia, classificou "que ótimo, o produto chegou quebrado" como positivo, mesmo sendo uma reclamação. Já nas frases mistas, não conseguiu identificar os dois sentimentos ao mesmo tempo.

Isso acontece porque o BoW considera as palavras individualmente, sem entender completamente a ordem e o sentido da frase. Duas melhorias possíveis seriam utilizar n-gramas para analisar combinações de palavras e testar modelos com embeddings para representar melhor o contexto.

O dataset também possui limitações, como a quantidade pequena de avaliações e a falta de exemplos neutros ou mistos. Um risco de usar esse modelo em uma loja seria deixar de identificar reclamações reais, prejudicando o atendimento aos clientes.

---

## Declaração de uso de IA

Marque uma opção e complete:

- [ ] Não usei IA generativa em nenhuma parte deste trabalho.
- [x] Usei IA generativa **somente** para: Usei IA generativa para auxiliar na criação e correção do código Python, explicar alguns conceitos e interpretar os resultados dos testes . (não é permitido gerar código ou relatório).