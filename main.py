import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from src.modelos import CONFIGURACOES, criar_pipeline
from sklearn.metrics import f1_score, make_scorer, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split


avaliacoes = pd.read_csv("data/avaliacoes.csv")
# print(avaliacoes.shape)


contagem_sentimentos = avaliacoes["sentimento"].value_counts(dropna=False)
# print(contagem_sentimentos)

textos = avaliacoes["texto"]
textos_vazios = textos.fillna("").str.strip().eq("").sum()
print(f"Quantidade de textos vazios: {textos_vazios}")

avaliacoes["qtd_palavras"] = textos.str.split().str.len()
media_palavras = avaliacoes.groupby("sentimento")["qtd_palavras"].mean()
print(media_palavras.round(2))

pasta_resultados = Path("resultados")
pasta_resultados.mkdir(exist_ok=True)
caminho_exploracao = pasta_resultados / "exploracao.txt"

with caminho_exploracao.open("w", encoding="utf-8") as arquivo:
    arquivo.write(f"total avaliacoes: {len(avaliacoes)}\n\n")

    arquivo.write("distribuiçao\n")
    for sentimento, quantidade in contagem_sentimentos.items():
        arquivo.write(f"{sentimento}: {quantidade}\n")

    arquivo.write("\nmedia palavars\n")
    for sentimento, media in media_palavras.items():
        arquivo.write(f"{sentimento}: {media:.2f}\n")

    for sentimento, textos_da_classe in avaliacoes.groupby("sentimento")["texto"]:
        arquivo.write(f"\nexemplos da classe {sentimento}\n")
        for numero, texto in enumerate(textos_da_classe.head(3), start=1):
            arquivo.write(f"{numero}. {texto}\n")

textos_treino, textos_teste, sentimentos_treino, sentimentos_teste = train_test_split(
    avaliacoes["texto"],
    avaliacoes["sentimento"],
    test_size=0.25,
    stratify=avaliacoes["sentimento"],
    random_state=42,
)

print(f"Treino: {len(textos_treino)} avaliações")
print(f"Teste: {len(textos_teste)} avaliações")

pipelines = {}
vocabularios = {}

for config, (classe_vetorizador, remover_stopwords) in CONFIGURACOES.items():
    pipeline = criar_pipeline(classe_vetorizador, remover_stopwords)
    pipelines[config] = pipeline

    vetorizador = clone(pipeline.named_steps["vetorizador"])
    vetorizador.fit(textos_treino)
    vocabularios[config] = len(vetorizador.vocabulary_)

    print(f"{config} — vocabulário de {vocabularios[config]} palavras")

validacao = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metricas = {
    "acc": "accuracy",
    "f1": make_scorer(f1_score, pos_label="positivo"),
}

linhas_comparacao = []

for config, pipeline in pipelines.items():
    pontuacoes = cross_validate(
        pipeline,
        textos_treino,
        sentimentos_treino,
        cv=validacao,
        scoring=metricas,
        error_score="raise",
    )

    acuracias = pontuacoes["test_acc"]
    valores_f1 = pontuacoes["test_f1"]
    classe_vetorizador, remover_stopwords = CONFIGURACOES[config]

    linhas_comparacao.append({
        "config": config,
        "stopwords": "removidas" if remover_stopwords else "mantidas",
        "vetorizador": classe_vetorizador.__name__,
        "vocabulario": vocabularios[config],
        "acc_cv_media": acuracias.mean(),
        "acc_cv_desvio": acuracias.std(),
        "f1_cv_media": valores_f1.mean(),
        "f1_cv_desvio": valores_f1.std(),
    })

comparacao = pd.DataFrame(linhas_comparacao)
comparacao.to_csv(pasta_resultados / "comparacao.csv", index=False, encoding="utf-8")

comparacao.to_csv(
    pasta_resultados / "comparacao.csv",
    columns=[
        "config", "stopwords", "vetorizador", "vocabulario",
        "acc_cv_media", "acc_cv_desvio", "f1_cv_media", "f1_cv_desvio",
    ],
    index=False,
    encoding="utf-8",
)

ranking_cv = comparacao.sort_values(
    by=["f1_cv_media", "acc_cv_media", "config"],
    ascending=[False, False, True],
)

config_vencedora = ranking_cv.iloc[0]["config"]

# print(ranking_cv[["config", "f1_cv_media", "acc_cv_media"]].to_string(index=False))
print(f"Configuração vencedora pela validação cruzada: {config_vencedora}")

modelo_final = pipelines[config_vencedora]
modelo_final.fit(textos_treino, sentimentos_treino)
previsoes_teste = modelo_final.predict(textos_teste)

classes = ["negativo", "positivo"]
matriz = confusion_matrix(sentimentos_teste, previsoes_teste, labels=classes)

pd.DataFrame(matriz, index=classes, columns=classes).to_csv(
    pasta_resultados / "matriz_confusao.csv",
    index_label="real",
    encoding="utf-8",
)

relatorio = classification_report(
    sentimentos_teste,
    previsoes_teste,
    labels=classes,
    digits=4,
    zero_division=0,
)
(pasta_resultados / "relatorio_classificacao.txt").write_text(
    relatorio,
    encoding="utf-8",
)

probabilidades_previstas = modelo_final.predict_proba(textos_teste).max(axis=1)

tabela_teste = pd.DataFrame({
    "texto": textos_teste.to_numpy(),
    "real": sentimentos_teste.to_numpy(),
    "previsto": previsoes_teste,
    "probabilidade": probabilidades_previstas,
})

erros = tabela_teste.loc[tabela_teste["real"] != tabela_teste["previsto"]]
erros.to_csv(pasta_resultados / "erros.csv", index=False, encoding="utf-8")

vetorizador_final = modelo_final.named_steps["vetorizador"]
classificador_final = modelo_final.named_steps["classificador"]

pesos = pd.DataFrame({
    "palavra": vetorizador_final.get_feature_names_out(),
    "coeficiente": classificador_final.coef_[0],
})

palavras_positivas = pesos.nlargest(10, "coeficiente").assign(
    sentimento=classificador_final.classes_[1]
)
palavras_negativas = pesos.nsmallest(10, "coeficiente").assign(
    sentimento=classificador_final.classes_[0]
)

palavras_influentes = pd.concat(
    [palavras_positivas, palavras_negativas],
    ignore_index=True,
)
palavras_influentes.to_csv(
    pasta_resultados / "palavras_influentes.csv",
    columns=["palavra", "sentimento", "coeficiente"],
    index=False,
    encoding="utf-8",
)

casos_dificeis = pd.read_csv("data/casos_dificeis.csv")
textos_casos = casos_dificeis["texto"]

indice_positivo = list(modelo_final.classes_).index("positivo")
probabilidades_casos = modelo_final.predict_proba(textos_casos)

resultado_casos = casos_dificeis[["texto", "tipo", "sentimento_esperado"]].rename(
    columns={"sentimento_esperado": "esperado"}
)
resultado_casos["previsto"] = modelo_final.predict(textos_casos)
resultado_casos["probabilidade_positivo"] = probabilidades_casos[:, indice_positivo]

resultado_casos.to_csv(
    pasta_resultados / "casos_dificeis_predicoes.csv",
    index=False,
    encoding="utf-8",
)