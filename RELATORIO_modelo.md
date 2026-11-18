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
tomei a decisão de padronizar as palavras, preservando a acentuação, ex: ÓTIMO, Ótimo e ótimo são a mesma palavra.

o código troca todas as pontuações e número por espaço, simplificando a contagem de palavras.

vocabulario com e sem, respectivamente: 425 e 413, sendo utilizado 20 stopwodrs.

a palavra "NÃO" não é incluida nos stopwords pois mudaria completamente a lógica, pense em: eu GOSTO e eu NÃO gosto, muda completamente.

## 2. BoW vs. TF-IDF

Com stopwords mantidas, o F1 médio foi 84,01% em A e 81,89% em B, com desvios de 7,55 e 9,00 pontos percentuais.

Com stopwords removidas, o F1 médio foi 85,06% em C e 84,62% em D, com desvios de 7,88 e 9,18 pontos percentuais.

O BoW teve a maior média nos dois pares. A distância entre o maior e o menor F1 médio foi de 3,17 pontos percentuais, pequena diante da variação entre os folds.

Escolhi C porque ela teve o maior F1 médio da classe positiva na validação cruzada.

O BoW conta quantas vezes cada palavra aparece. O TF-IDF dá mais peso a palavras que aparecem no texto e são menos comuns no conjunto de treino.

a vantagem foi modesta nesse experimento. A configuracao C foi escolhida por ter o maior F1 medio da classe positiva na validacao cruzada.

o BoW conta quantas vezes cada palavra aparece. O TF-IDF pondera as ocorrencias, dando mais peso a palavras que aparecem no texto e sao menos comuns no conjunto de treino.

## 3. Avaliação no teste

A matriz foi [[25, 8], [3, 29]], com classes reais nas linhas e previstas nas colunas, na ordem negativo, positivo.

A diagonal soma 54 acertos em 65 avaliações, o que dá 83,08% de acurácia.

Na classe negativa, 25 de 28 previsões estavam corretas (89,29% de precisão) e 25 de 33 casos reais foram encontrados (75,76% de recall). O F1 foi 81,97%.

Na classe positiva, foram 29 de 37 previsões corretas (78,38% de precisão) e 29 de 32 casos reais encontrados (90,62% de recall). O F1 foi 84,06%.

Para uma loja que usa o modelo para encaminhar reclamações, o erro mais grave é marcar uma reclamação como elogio. Isso ocorreu 8 vezes e poderia deixar clientes sem atendimento.

O teste tinha 33 negativas e 32 positivas. Prever sempre a classe majoritária daria só 50,77% de acurácia. Quando fraudes são raras, prever sempre que a transação é legítima pode dar acurácia alta e deixar passar todas as fraudes.

Não lembro qual o exemplo de fraude da Aula 8 e não encontrei no aula.

## 4. Validação cruzada vs. divisão única

A validação cruzada usou as 195 avaliações de treino em cinco rodadas. Em cada uma, 156 avaliações foram usadas para ajuste e 39 para validação.

As quatro configurações passaram pelas mesmas divisões. A média reduz a influência de uma divisão específica, e o desvio mostra quanto o desempenho variou.

As 65 avaliações de teste ficaram reservadas para a avaliação final. Escolher o modelo olhando o teste poderia favorecer a configuração que teve sorte naquela amostra.

Haveria vazamento se o vocabulário ou o IDF fosse calculado usando textos da validação ou do teste.

A Pipeline ajustou o vetorizador e a Regressão Logística só com a parte de treino de cada fold.

## 5. Análise de erros e palavras influentes

“Ficou feio na sala, a cor é bem diferente da foto” recebeu positivo com 64,14%. Feio ficou fora do vocabulário, e bem, que intensifica diferente nessa frase, teve peso positivo de +0,944.

“Cheguei animado, mas me arrependi, tudo estava sujo” recebeu positivo com 66,28%. O BoW trata as palavras separadamente. Arrependi ficou fora do vocabulário e tudo teve peso positivo de +0,706.

“O sabor é maravilhoso e as porções são generosas” recebeu negativo com 55,52%. Maravilhoso contribuiu a favor de positivo, mas generosas ficou fora do vocabulário e palavras como sabor e porções tiveram peso negativo.

Bastante (+2,196) é um intensificador, e dinheiro (−0,595) pode aparecer tanto em elogios quanto em reclamações.

Esses pesos mostram que o modelo aprendeu associações entre palavras e rótulos nos dados de treino. O sentido de cada palavra depende da frase em que aparece.


## 6. Casos difíceis e limitações
*Com base em `casos_dificeis_predicoes.csv`, como o modelo se saiu em negação, ironia e frases mistas? Explique por que BoW/TF-IDF têm essa limitação, proponha duas melhorias (cite técnicas vistas na Aula 11) e discuta duas limitações do dataset e um risco de usar este modelo em produção.*

O modelo acertou 2 de 4 casos de negação, 2 de 3 casos de ironia e 7 de 10 frases com rótulo esperado positivo ou negativo.

As duas frases mistas receberam positivo. A neutra, “Chegou ontem”, recebeu negativo, com 49,64% de probabilidade de positivo. Analisei esses três casos separadamente porque o modelo só prevê duas classes.

Usando palavras isoladas, BoW e TF-IDF perdem a ordem e o contexto. “Que ótimo, o produto chegou quebrado” recebeu positivo com 83,53% de probabilidade de positivo.

Uma melhoria seria testar bigramas para representar sequências como não gostei. Outra seria testar um léxico com regras de negação para ajustar a polaridade das palavras negadas.

A amostra tem 260 avaliações, e parte das frases foi gerada por padrões. Isso pode favorecer resultados mais altos do que os obtidos com avaliações reais e mais variadas.

Em produção, uma reclamação marcada como elogio poderia ficar sem atendimento. No teste, 8 das 33 avaliações negativas receberam o rótulo positivo.

*---*

## Declaração de uso de IA

Marque uma opção e complete:

- [ ] Não usei IA generativa em nenhuma parte deste trabalho.
- [x] Usei IA generativa **somente** para: pesquisar erros e para resolução de problemas, sem geração de códigos ou o relatório. Também utilizei para formatação, pois escrevi o relatório a mil por hora e parecia que português era minha 4° lingua.
