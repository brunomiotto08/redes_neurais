"""Carga, limpeza e divisão do dataset da NBA.

Pandas organiza a tabela. O StandardScaler só padroniza as entradas,
e é ajustado exclusivamente no treino. A divisão estratificada é manual.
"""

import math
import random
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

POSICOES = ["PG", "SG", "SF", "PF", "C"]
FEATURES_NUMERICAS = [
    "Age",
    "G",
    "GS",
    "MP",
    "FG%",
    "3P%",
    "2P%",
    "eFG%",
    "FT%",
    "ORB",
    "DRB",
    "TRB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
]
FEATURE_PONTOS = "PTS"
VARIAVEIS_VOLUME = ["FG", "FGA", "3P", "3PA", "2P", "2PA", "FT", "FTA", "PTS"]
PARES_PERCENTUAL = [
    ("FG%", "FGA"),
    ("3P%", "3PA"),
    ("2P%", "2PA"),
    ("eFG%", "FGA"),
    ("FT%", "FTA"),
]
COLUNAS_NUMERICAS = FEATURES_NUMERICAS + VARIAVEIS_VOLUME
SEMENTE_DIVISAO = 42


def _para_numero(serie):
    if serie.dtype == object:
        serie = serie.astype(str).str.strip().str.replace(",", ".", regex=False)
        serie = serie.replace({"": None, "nan": None, "None": None})
    return pd.to_numeric(serie, errors="coerce")


def _posicao_base(valor):
    texto = str(valor).strip().upper()
    if texto in {"", "NAN", "NONE"}:
        return None
    texto = texto.split("-")[0].strip()
    if texto not in POSICOES:
        return None
    return texto


def _cortar_indices(indices, gerador):
    indices = list(indices)
    gerador.shuffle(indices)
    n = len(indices)
    if n < 3:
        raise RuntimeError(f"Classe com apenas {n} exemplos, insuficiente para a divisão.")
    n_teste = max(1, round(0.2 * n))
    n_val = max(1, round(0.2 * n))
    while n_teste + n_val >= n:
        if n_teste >= n_val and n_teste > 1:
            n_teste -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break
    teste = indices[:n_teste]
    validacao = indices[n_teste : n_teste + n_val]
    treino = indices[n_teste + n_val :]
    return treino, validacao, teste


def _pearson(xs, ys):
    n = len(xs)
    media_x = sum(xs) / n
    media_y = sum(ys) / n
    numerador = 0.0
    soma_x = 0.0
    soma_y = 0.0
    for x, y in zip(xs, ys):
        dx = x - media_x
        dy = y - media_y
        numerador += dx * dy
        soma_x += dx * dx
        soma_y += dy * dy
    if soma_x == 0.0 or soma_y == 0.0:
        return 0.0
    return numerador / math.sqrt(soma_x * soma_y)


def _matriz(df, colunas):
    return [
        [float(valor) for valor in linha]
        for linha in df[colunas].itertuples(index=False, name=None)
    ]


def _padronizar(treino, validacao, teste):
    scaler = StandardScaler()
    treino_padrao = scaler.fit_transform(treino).tolist()
    validacao_padrao = scaler.transform(validacao).tolist()
    teste_padrao = scaler.transform(teste).tolist()
    return treino_padrao, validacao_padrao, teste_padrao


def _preencher_mediana(treino, validacao, teste, colunas):
    medianas = {}
    for coluna in colunas:
        valores = treino[coluna].dropna().sort_values().tolist()
        if not valores:
            raise RuntimeError(f"A coluna {coluna} está vazia no treino.")
        meio = len(valores) // 2
        if len(valores) % 2 == 1:
            mediana = float(valores[meio])
        else:
            mediana = float((valores[meio - 1] + valores[meio]) / 2.0)
        medianas[coluna] = mediana
        treino[coluna] = treino[coluna].fillna(mediana)
        validacao[coluna] = validacao[coluna].fillna(mediana)
        teste[coluna] = teste[coluna].fillna(mediana)
    return medianas


def carregar_e_limpar(caminho):
    df = pd.read_csv(caminho, encoding="utf-8")
    df.columns = [str(coluna).strip() for coluna in df.columns]
    resumo = {"linhas_brutas": int(len(df)), "colunas_brutas": list(df.columns)}

    if "FG%.1" in df.columns:
        df = df.drop(columns=["FG%.1"])
        resumo["coluna_duplicada_removida"] = "FG%.1"
    else:
        resumo["coluna_duplicada_removida"] = None

    df["Player"] = df["Player"].astype(str).str.strip()
    df.loc[df["Player"].isin(["", "nan", "None"]), "Player"] = pd.NA
    df = df.dropna(subset=["Player"])
    df = df[df["Performance"].isin(["Good", "Bad"])].copy()
    resumo["linhas_apos_remover_invalidas"] = int(len(df))

    for coluna in COLUNAS_NUMERICAS + ["Pos", "Tm"]:
        if coluna not in df.columns:
            raise RuntimeError(f"Coluna ausente no CSV: {coluna}")
    for coluna in COLUNAS_NUMERICAS:
        df[coluna] = _para_numero(df[coluna])
    df["Tm"] = df["Tm"].astype(str).str.strip()

    repetidos = int(df["Player"].duplicated().sum())
    resumo["jogadores_com_mais_de_uma_linha"] = int(df.loc[df["Player"].duplicated(keep=False), "Player"].nunique())
    resumo["linhas_extras_de_jogadores_repetidos"] = repetidos

    tem_tot = set(df.loc[df["Tm"] == "TOT", "Player"])
    parte_tot = df.loc[df["Player"].isin(tem_tot) & df["Tm"].eq("TOT")].drop_duplicates("Player", keep="first")
    parte_resto = (
        df.loc[~df["Player"].isin(tem_tot)]
        .sort_values(["Player", "G"], ascending=[True, False], kind="mergesort")
        .drop_duplicates("Player", keep="first")
    )
    n_antes = len(df)
    df = pd.concat([parte_tot, parte_resto], ignore_index=True)
    resumo["linhas_repetidas_removidas"] = int(n_antes - len(df))
    resumo["jogadores_mantidos_pela_linha_TOT"] = int(len(tem_tot))

    resumo["posicoes_hibridas"] = int(df["Pos"].astype(str).str.contains("-", na=False).sum())
    df["PosBase"] = df["Pos"].map(_posicao_base)
    sem_posicao = int(df["PosBase"].isna().sum())
    df = df.dropna(subset=["PosBase"]).copy()
    resumo["linhas_sem_posicao_removidas"] = sem_posicao

    preenchidos = 0
    for percentual, tentativas in PARES_PERCENTUAL:
        mascara = df[percentual].isna() & (df[tentativas].fillna(0) == 0)
        preenchidos += int(mascara.sum())
        df.loc[mascara, percentual] = 0.0
    resumo["percentuais_sem_tentativa_preenchidos_com_zero"] = preenchidos

    df["y"] = (df["Performance"] == "Good").astype(int)
    for posicao in POSICOES:
        df[f"Pos_{posicao}"] = (df["PosBase"] == posicao).astype(float)

    resumo["linhas_finais"] = int(len(df))
    resumo["good"] = int((df["y"] == 1).sum())
    resumo["bad"] = int((df["y"] == 0).sum())
    resumo["pts_good_min"] = float(df.loc[df["y"] == 1, "PTS"].min())
    resumo["pts_good_max"] = float(df.loc[df["y"] == 1, "PTS"].max())
    resumo["pts_bad_min"] = float(df.loc[df["y"] == 0, "PTS"].min())
    resumo["pts_bad_max"] = float(df.loc[df["y"] == 0, "PTS"].max())
    resumo["good_com_pts_alto"] = int(((df["y"] == 1) & (df["PTS"] >= 20)).sum())
    resumo["bad_com_pts_maior_ou_igual_20"] = int(((df["y"] == 0) & (df["PTS"] >= 20)).sum())

    def _casos(mascara):
        recorte = df.loc[mascara, ["Player", "Tm", "Pos", "PTS", "MP"]].sort_values("PTS", ascending=False)
        registros = []
        for linha in recorte.itertuples(index=False):
            registros.append(
                {
                    "Player": str(linha.Player),
                    "Tm": str(linha.Tm),
                    "Pos": str(linha.Pos),
                    "PTS": float(linha.PTS),
                    "MP": None if linha.MP != linha.MP else float(linha.MP),
                }
            )
        return registros

    resumo["excecoes_bad_pontuadores"] = _casos((df["y"] == 0) & (df["PTS"] >= 20))
    resumo["excecoes_good_poucos_pontos"] = _casos((df["y"] == 1) & (df["PTS"] < 20.5))
    resumo["good_no_intervalo_de_pts_do_bad"] = _casos(
        (df["y"] == 1) & (df["PTS"] <= resumo["pts_bad_max"])
    )
    medias = {}
    for classe, nome in ((1, "Good"), (0, "Bad")):
        bloco = df.loc[df["y"] == classe, ["PTS", "MP", "AST", "TRB", "FG%", "Age", "G"]]
        medias[nome] = {coluna: round(float(bloco[coluna].mean()), 3) for coluna in bloco.columns}
    resumo["medias_por_classe"] = medias
    return df, resumo


def preparar_dados(caminho=None):
    if caminho is None:
        caminho = Path(__file__).resolve().parents[1] / "data" / "nba_players.csv"
    df, resumo = carregar_e_limpar(caminho)
    nomes = FEATURES_NUMERICAS + [f"Pos_{posicao}" for posicao in POSICOES]
    nomes_pts = FEATURES_NUMERICAS + [FEATURE_PONTOS] + [f"Pos_{posicao}" for posicao in POSICOES]
    colunas_imputacao = FEATURES_NUMERICAS + [FEATURE_PONTOS]

    gerador = random.Random(SEMENTE_DIVISAO)
    idx_treino, idx_val, idx_teste = [], [], []
    for classe in (0, 1):
        treino, validacao, teste = _cortar_indices(df.index[df["y"] == classe], gerador)
        idx_treino.extend(treino)
        idx_val.extend(validacao)
        idx_teste.extend(teste)

    treino = df.loc[idx_treino].copy()
    validacao = df.loc[idx_val].copy()
    teste = df.loc[idx_teste].copy()

    nomes_jogadores = {
        "treino": treino["Player"].tolist(),
        "validacao": validacao["Player"].tolist(),
        "teste": teste["Player"].tolist(),
    }
    if not set(nomes_jogadores["treino"]).isdisjoint(nomes_jogadores["validacao"]):
        raise RuntimeError("O mesmo jogador caiu em treino e validação.")
    if not set(nomes_jogadores["treino"]).isdisjoint(nomes_jogadores["teste"]):
        raise RuntimeError("O mesmo jogador caiu em treino e teste.")
    if not set(nomes_jogadores["validacao"]).isdisjoint(nomes_jogadores["teste"]):
        raise RuntimeError("O mesmo jogador caiu em validação e teste.")

    medianas = _preencher_mediana(treino, validacao, teste, colunas_imputacao)
    resumo["medianas_treino"] = {chave: round(valor, 6) for chave, valor in medianas.items()}
    resumo["divisao"] = {}
    for nome, parte in ("treino", treino), ("validacao", validacao), ("teste", teste):
        resumo["divisao"][nome] = {
            "n": int(len(parte)),
            "good": int((parte["y"] == 1).sum()),
            "bad": int((parte["y"] == 0).sum()),
        }
    resumo["semente_divisao"] = SEMENTE_DIVISAO
    resumo["features_modelo"] = nomes
    resumo["features_diagnostico_pts"] = nomes_pts
    resumo["variaveis_volume_fora_do_modelo"] = VARIAVEIS_VOLUME
    resumo["variaveis_nao_usadas"] = ["Player", "Tm", "FG%.1"]

    y_treino = [int(valor) for valor in treino["y"].tolist()]
    y_val = [int(valor) for valor in validacao["y"].tolist()]
    y_teste = [int(valor) for valor in teste["y"].tolist()]

    correlacoes = {}
    for coluna in FEATURES_NUMERICAS + [FEATURE_PONTOS]:
        correlacoes[coluna] = round(_pearson(treino[coluna].astype(float).tolist(), y_treino), 4)
    resumo["correlacao_com_good_no_treino"] = correlacoes

    for parte, rotulos in (treino, y_treino), (validacao, y_val), (teste, y_teste):
        if parte[nomes_pts].isna().any().any():
            raise RuntimeError("Ainda há valor ausente depois da mediana do treino.")
        if len(parte) != len(rotulos):
            raise RuntimeError("Rótulos e linhas ficaram desalinhados.")

    x_treino = _matriz(treino, nomes)
    x_val = _matriz(validacao, nomes)
    x_teste = _matriz(teste, nomes)
    x_treino_pts = _matriz(treino, nomes_pts)
    x_val_pts = _matriz(validacao, nomes_pts)
    x_teste_pts = _matriz(teste, nomes_pts)

    x_treino, x_val, x_teste = _padronizar(x_treino, x_val, x_teste)
    x_treino_pts, x_val_pts, x_teste_pts = _padronizar(x_treino_pts, x_val_pts, x_teste_pts)

    return {
        "df_eda": df,
        "nomes": nomes,
        "nomes_pts": nomes_pts,
        "X_treino": x_treino,
        "y_treino": y_treino,
        "X_val": x_val,
        "y_val": y_val,
        "X_teste": x_teste,
        "y_teste": y_teste,
        "X_treino_pts": x_treino_pts,
        "X_val_pts": x_val_pts,
        "X_teste_pts": x_teste_pts,
        "resumo": resumo,
    }
