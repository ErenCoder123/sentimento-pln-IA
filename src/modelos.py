from functools import partial

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessamento import preprocessar_texto


CONFIGURACOES = {
    "A": (CountVectorizer, False),
    "B": (TfidfVectorizer, False),
    "C": (CountVectorizer, True),
    "D": (TfidfVectorizer, True),
}


def criar_pipeline(classe_vetorizador, remover_stopwords):
    preparar_texto = partial(
        preprocessar_texto,
        remover_stopwords=remover_stopwords,
    )
    vetorizador = classe_vetorizador(
        preprocessor=preparar_texto,
        tokenizer=str.split,
        token_pattern=None,
    )
    return Pipeline([
        ("vetorizador", vetorizador),
        ("classificador", LogisticRegression(max_iter=1000, random_state=42)),
    ])