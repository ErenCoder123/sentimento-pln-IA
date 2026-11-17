import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

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