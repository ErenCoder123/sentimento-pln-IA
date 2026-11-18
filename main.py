import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from src.modelos import CONFIGURACOES, criar_pipeline
from sklearn.metrics import f1_score, make_scorer
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