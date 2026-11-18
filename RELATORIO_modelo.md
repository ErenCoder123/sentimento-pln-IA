# RELATORIO — Análise de Sentimento com PLN

**Nome: Miguel**
**Repositório: https://github.com/ErenCoder123/sentimento-pln-IA** 
**Data:17/11/2026** 

> Responda cada pergunta com **4 a 12 linhas**, com suas palavras e usando números dos arquivos de `resultados/`. Apague as instruções em itálico ao entregar.


| Config | Stopwords | Vetorizador | Vocabulário | Acurácia CV (média ± desvio) | F1 CV (média ± desvio) |
|---|---|---|---|---|---|
| A | mantidas | BoW | 425| 84,10% ± 7,14| 84,01% ± 7,55|
| B | mantidas | TF-IDF | 425| 82,05% ± 9,17| 81,89% ± 9,00|
| C | removidas | BoW | 413| 85,64% ± 7,36| 85,46% ± 7,85|
| D | removidas | TF-IDF | 413| 83,08% ± 8,82| 82,88% ± 8,89|

**Configuração escolhida:** C, Bag of Words com remoção de stopwords. | **Acurácia no teste:** 83,08% | **F1 no teste:** 83,58%

---

## 1. Pré-processamento
*Quais decisões você tomou (minúsculas, pontuação, números, stopwords)? Qual o tamanho do vocabulário com e sem remoção de stopwords? O que você fez com a palavra "não" e por quê?*

tomei a decisão de padronizar as palavras, preservando a acentuação, ex: ÓTIMO, Ótimo e ótimo são a mesma palavra.

o código troca todas as pontuações e número por espaço, simplificando a contagem de palavras.

vocabulario com e sem, respectivamente: 425 e 413, sendo utilizado 20 stopwodrs.

a palavra "NÃO" não é incluida nos stopwords pois mudaria completamente a lógica, pense em: eu GOSTO e eu NÃO gosto, muda completamente.

## 2. BoW vs. TF-IDF
*Compare as quatro configurações levando em conta o desvio padrão. As diferenças são rele**vantes? Qual você escolheu e com base em quê? Explique a diferença conceitual entre as duas representações**.*

com stopwords mantidas, o F1 médio foi 84,01% no BoW (A) e 81,89% no TF-IDF (B). Os desvios foram 7,55 e 9,00 pontos percentuais.

com stopwords removidas, o F1 médio foi 85,46% no BoW (C) e 82,88% no TF-IDF (D). Os desvios foram 7,85 e 8,89 pontos percentuais.

o BoW teve a maior media nos dois pares. A maior diferença entre as medias foi de 3,57 pontos.

a vantagem foi modesta nesse experimento. A configuracao C foi escolhida por ter o maior F1 medio da classe positiva na validacao cruzada.

o BoW conta quantas vezes cada palavra aparece. O TF-IDF pondera as ocorrencias, dando mais peso a palavras que aparecem no texto e sao menos comuns no conjunto de treino.

## 3. Avaliação no teste
*Mostre a matriz de confusão e interprete precisão, recall e F1 de cada classe. Para uma loja, qual erro é mais grave: classificar uma reclamação como elogio ou o contrário? Por que a acurácia é menos enganosa aqui do que no exemplo de fraude da Aula 8?*

na ordem negativo, positivo, a matriz foi [[26, 7], [4, 28]], com as classes reais nas linhas e as previstas nas colunas.
a diagonal soma 54 acertos em 65 avaliacoes, o que dá 83,08% de acuracia.

na classe negativa, a precisao foi 86,67% (26 de 30 previsoes negativas), o recall foi 78,79% (26 de 33 casos reais) e o F1 foi 82,54%.

na classe positiva, a precisao foi 80,00% (28 de 35 previsoss positivas), o recall foi 87,50% (28 de 32 casos reais) e o F1 foi 83,58%. O F1 combina precisao e recall.

para uma loja que usa o modelo para encaminhar reclamações, classificar uma reclamação como elogio é mais grave. Isso aconteceu 7 vezes no teste e poderia deixar clientes sem atendimento.

nao lembro qual o exemplo de fraude da Aula 8 e não encontrei no aula.

## 4. Validação cruzada vs. divisão única
*Por que a validação cruzada é mais confiável do que um único conjunto de teste para comparar configurações? Por que a escolha não pode ser feita olhando o teste? O que seria vazamento de dados neste trabalho e como você o evitou?*

a validacao cruzada dividiu as 195 avaliacoes de treino em cinco folds, com 156 exemplos para ajuste e 39 para validação a cada rodada.

as quatro configuracees passaram pelas mesmas divisoes. Isso permitiu comparar as medias e observar quanto os resultados variaram entre os folds.

as 65 avaliações de teste ficaram reservadas para medir o desempenho da configuração escolhida. uasr elas para decidir a vencedora favoreceria a configuracao que se saiu melhor naquela amostra específica.

haveria vazamento se o vocabulario ou o IDF fosse ajustado usando textos da validaçao ou do teste.

a pipeline evitou isso ao ajustar o vetorizador e a regressao apenas com os exemplos de treino de cada fold.

## 5. Análise de erros e palavras influentes
*Escolha 3 textos de `erros.csv` e explique por que o modelo pode ter errado. Observe as 10 palavras mais influentes de cada classe: há alguma que não carrega sentimento por si só (ex.: palavra funcional ou de domínio)? O que isso diz sobre o que o modelo realmente aprendeu?*

“Ficou feio na sala, a cor é bem diferente da foto” recebeu positivo com 62,22%. Bem intensifica diferente nessa frase e tem peso positivo de +0,954 no modelo.

“Cheguei animado, mas me arrependi, tudo estava sujo” recebeu positivo com 64,26%. O mas muda o sentido entre as duas partes, e o BoW conta as palavras separadamente.

“Ficou lindo na sala, combinou perfeitamente com os móveis” recebeu negativo com 57,96%. Uma hipótese é que perfeitamente e perfeito são tratados como palavras distintas. O peso de +0,790 pertence a perfeito.

Entre os termos influentes, bastante (+2,257) é um intensificador, e dinheiro (−0,673) pode aparecer tanto em elogios quanto em reclamações.

Esses pesos mostram associações aprendidas nos dados de treino. Eles ajudam a levantar hipóteses sobre os erros, pois o modelo combina os pesos de várias palavras para fazer uma previsão.

## 6. Casos difíceis e limitações
*Com base em `casos_dificeis_predicoes.csv`, como o modelo se saiu em negação, ironia e frases mistas? Explique por que BoW/TF-IDF têm essa limitação, proponha duas melhorias (cite técnicas vistas na Aula 11) e discuta duas limitações do dataset e um risco de usar este modelo em produção.*

O modelo acertou 2 de 4 casos de negação, 2 de 3 de ironia e 7 de 10 entre as frases com rótulo esperado positivo ou negativo.

As duas frases mistas receberam positivo. A frase neutra, “Chegou ontem”, recebeu negativo, com 47,30% de probabilidade de positivo. Esses três casos foram analisados qualitativamente porque o modelo só prevê duas classes.

Com palavras isoladas, BoW e TF-IDF perdem a ordem e o contexto. Por exemplo, “Que ótimo, o produto chegou quebrado” recebeu positivo com 81,93% de probabilidade de positivo.

Uma melhoria seria testar bigramas, que permitem representar sequências como não gostei. Outra seria testar um léxico com tratamento de negação, que ajusta a polaridade das palavras no trecho negado.

A amostra tem apenas 260 avaliações, e parte delas foi gerada por padrões. Frases parecidas no treino e no teste podem favorecer resultados mais altos do que os obtidos com avaliações reais.

Em produção, um risco seria deixar reclamações sem atendimento. No teste, 7 das 33 avaliações negativas foram classificadas como positivas.

*---*

## Declaração de uso de IA

Marque uma opção e complete:

- [ ] Não usei IA generativa em nenhuma parte deste trabalho.
- [x] Usei IA generativa **somente** para: pesquisar erros e para resolução de problemas, sem geração de códigos ou o relatório. (não é permitido gerar código ou relatório).
