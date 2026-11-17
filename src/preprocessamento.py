from unicodedata import normalize

STOPWORDS = {"a", "o", "as", "os", "um", "uma", "uns", "umas"}


def preprocessar_texto(texto: str, remover_stopwords: bool = False) -> str:
    texto_normalizado = normalize("NFC", texto.lower())
    caracteres = [
        caractere if caractere.isalpha() else " "
        for caractere in texto_normalizado
    ]
    palavras = "".join(caracteres).split()

    if remover_stopwords:
        palavras = [palavra for palavra in palavras if palavra not in STOPWORDS]

    return " ".join(palavras)