"""Treino, comparação de arquiteturas e figuras do trabalho.

A taxa de aprendizado e o número de épocas saem de um piloto na validação.
Cada arquitetura roda 5 vezes, mudando só a semente dos pesos.
O limiar de decisão também sai da validação, pelo melhor F1 de Good.
"""

import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metricas import (
    area_sob_roc,
    calcular_metricas,
    curva_roc,
    importancia_por_permutacao,
    melhor_limiar,
)
from src.mlp import RedeMLP, pesos_de_classe, verificar_aprendizado, verificar_gradiente
from src.preparacao import preparar_dados

RAIZ = Path(__file__).resolve().parents[1]
FIGURAS = RAIZ / "figuras"
RESULTADOS = RAIZ / "resultados"

EPOCAS_PILOTO = 300
TAXAS = [0.01, 0.05, 0.1]
SEMENTES = [1, 2, 3, 4, 5]
ARQUITETURAS = [
    ("8 neurônios", (8,)),
    ("16 neurônios", (16,)),
    ("16 e 8 neurônios", (16, 8)),
]
CAMPOS = [
    "acuracia",
    "erro",
    "recall",
    "precisao",
    "especificidade",
    "f1",
    "auc",
    "acuracia_balanceada",
]
COR_BAD = "#9aa0a6"
COR_GOOD = "#1d4e89"
CORES = {
    "8 neurônios": "#1d4e89",
    "16 neurônios": "#b45309",
    "16 e 8 neurônios": "#2f6f4e",
    "MLP com PTS": "#6b3fa0",
}


def _estilo():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def _salvar_figura(fig, nome):
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURAS / nome, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _media_movel(valores, janela=15):
    suavizada = []
    for indice in range(len(valores)):
        inicio = max(0, indice - janela + 1)
        trecho = valores[inicio : indice + 1]
        suavizada.append(sum(trecho) / len(trecho))
    return suavizada


def escolher_hiperparametros(pilotos):
    """Taxa com menor perda final de validação.

    As épocas são o ponto de menor perda suavizada dessa taxa.
    Se a curva ainda caía no fim do piloto, ficam todas as épocas.
    """
    vencedor = min(pilotos, key=lambda item: item["historico"][-1]["perda_validacao"])
    suavizada = _media_movel([ponto["perda_validacao"] for ponto in vencedor["historico"]])
    indice = min(range(len(suavizada)), key=lambda i: suavizada[i])
    epocas = indice + 1
    if indice >= int(0.9 * (len(suavizada) - 1)):
        epocas = len(suavizada)
    if epocas < 40:
        epocas = 40
    return vencedor["taxa"], epocas


def _resumir(execucoes):
    medias = {}
    desvios = {}
    for campo in CAMPOS:
        valores = [item["metricas_teste"][campo] for item in execucoes]
        medias[campo] = statistics.mean(valores)
        desvios[campo] = statistics.stdev(valores) if len(valores) > 1 else 0.0
    f1_validacao = statistics.mean(item["metricas_validacao"]["f1"] for item in execucoes)
    return medias, desvios, f1_validacao


def _treinar(amostras, rotulos, amostras_val, rotulos_val, ocultas, taxa, epocas, semente, pesos):
    rede = RedeMLP(
        [len(amostras[0]), *ocultas, 1],
        semente=semente,
        pesos_classe=pesos,
    )
    historico = rede.treinar(
        amostras,
        rotulos,
        amostras_val,
        rotulos_val,
        taxa=taxa,
        epocas=epocas,
        a_cada=50,
    )
    return rede, historico


def _avaliar_execucao(rede, historico, X_val, y_val, X_teste, y_teste, semente):
    escolha = melhor_limiar(y_val, rede.prever_proba(X_val))
    limiar = escolha["limiar"]
    probabilidades = rede.prever_proba(X_teste)
    previsto = [1 if probabilidade >= limiar else 0 for probabilidade in probabilidades]
    metricas = calcular_metricas(y_teste, previsto)
    roc = curva_roc(y_teste, probabilidades)
    metricas["auc"] = area_sob_roc(roc)
    return {
        "semente": semente,
        "limiar": limiar,
        "metricas_validacao": escolha["metricas"],
        "metricas_teste": metricas,
        "roc": [{"fpr": ponto["fpr"], "tpr": ponto["tpr"]} for ponto in roc],
        "historico": historico,
        "rede": rede,
    }


def _rodar_grupo(nome, ocultas, taxa, epocas, pesos, X_treino, y_treino, X_val, y_val, X_teste, y_teste):
    print(f"\nArquitetura {nome} | taxa {taxa} | épocas {epocas}", flush=True)
    execucoes = []
    for semente in SEMENTES:
        print(f"  semente {semente}", flush=True)
        rede, historico = _treinar(
            X_treino, y_treino, X_val, y_val, ocultas, taxa, epocas, semente, pesos
        )
        execucoes.append(_avaliar_execucao(rede, historico, X_val, y_val, X_teste, y_teste, semente))
    medias, desvios, f1_validacao = _resumir(execucoes)
    return {
        "nome": nome,
        "ocultas": list(ocultas),
        "medias": medias,
        "desvios": desvios,
        "f1_validacao_media": f1_validacao,
        "execucoes": execucoes,
    }


def _baseline(y_teste):
    previsto = [0 for _ in y_teste]
    metricas = calcular_metricas(y_teste, previsto)
    metricas["auc"] = area_sob_roc(curva_roc(y_teste, [0.0 for _ in y_teste]))
    return metricas


def _plotar_eda(df, resumo):
    fig, ax = plt.subplots(figsize=(6, 4))
    valores = [resumo["bad"], resumo["good"]]
    barras = ax.bar(["Bad", "Good"], valores, color=[COR_BAD, COR_GOOD])
    for barra, valor in zip(barras, valores):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height(),
            str(valor),
            ha="center",
            va="bottom",
        )
    ax.set_ylabel("Jogadores")
    ax.set_ylim(0, max(valores) * 1.12)
    ax.set_title("Classes depois da limpeza")
    ax.grid(axis="x", visible=False)
    _salvar_figura(fig, "distribuicao_classes.png")

    fig, eixos = plt.subplots(1, 2, figsize=(10, 4))
    for eixo, coluna, titulo in (
        (eixos[0], "PTS", "Pontos por jogo"),
        (eixos[1], "MP", "Minutos por jogo"),
    ):
        bad = df.loc[df["y"] == 0, coluna].dropna().tolist()
        good = df.loc[df["y"] == 1, coluna].dropna().tolist()
        eixo.hist(bad, bins=20, alpha=0.75, label="Bad", color=COR_BAD)
        eixo.hist(good, bins=15, alpha=0.8, label="Good", color=COR_GOOD)
        eixo.set_title(titulo)
        eixo.set_ylabel("Jogadores")
        eixo.legend()
    eixos[0].axvline(
        resumo["pts_good_min"],
        color="#b45309",
        linestyle="--",
        label=f"menor PTS Good ({resumo['pts_good_min']:.1f})",
    )
    eixos[0].legend()
    fig.suptitle("A classe Good acompanha o volume de pontos")
    _salvar_figura(fig, "hist_pts_mp.png")


def _plotar_piloto(pilotos):
    fig, ax = plt.subplots(figsize=(7, 4))
    for piloto in pilotos:
        epocas = [ponto["epoca"] for ponto in piloto["historico"]]
        perda = [ponto["perda_validacao"] for ponto in piloto["historico"]]
        ax.plot(epocas, perda, label=f"taxa {piloto['taxa']}")
    ax.set_xlabel("Época")
    ax.set_ylabel("Perda na validação")
    ax.set_title("Piloto da taxa de aprendizado (16 neurônios)")
    ax.legend()
    _salvar_figura(fig, "curva_perda_piloto.png")


def _plotar_perda(historico, titulo):
    fig, ax = plt.subplots(figsize=(7, 4))
    epocas = [ponto["epoca"] for ponto in historico]
    ax.plot(epocas, [ponto["perda_treino"] for ponto in historico], label="Treino", color=COR_GOOD)
    ax.plot(
        epocas,
        [ponto["perda_validacao"] for ponto in historico],
        label="Validação",
        color="#b45309",
    )
    ax.set_xlabel("Época")
    ax.set_ylabel("Entropia cruzada ponderada")
    ax.set_title(titulo)
    ax.legend()
    _salvar_figura(fig, "curva_perda_melhor.png")


def _plotar_matriz(metricas, titulo):
    grade = [
        [metricas["tn"], metricas["fp"]],
        [metricas["fn"], metricas["tp"]],
    ]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(grade, cmap="Blues")
    ax.set_xticks([0, 1], ["Previsto Bad", "Previsto Good"])
    ax.set_yticks([0, 1], ["Real Bad", "Real Good"])
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(grade[i][j]), ha="center", va="center", fontsize=16, color="black")
    ax.set_title(titulo)
    _salvar_figura(fig, "matriz_confusao.png")


def _plotar_roc(curvas):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#b0b0b0", label="Aleatório")
    for curva in curvas:
        pontos = sorted(curva["roc"], key=lambda ponto: (ponto["fpr"], ponto["tpr"]))
        ax.plot(
            [ponto["fpr"] for ponto in pontos],
            [ponto["tpr"] for ponto in pontos],
            color=curva["cor"],
            linestyle=curva.get("estilo", "-"),
            label=curva["nome"],
        )
    ax.set_xlabel("FPR (1 − especificidade)")
    ax.set_ylabel("TPR (recall de Good)")
    ax.set_title("Curva ROC no teste")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    _salvar_figura(fig, "roc_arquiteturas.png")


def _plotar_barras(grupos):
    nomes_metricas = ["acuracia", "recall", "precisao", "especificidade"]
    rotulos = ["Acurácia", "Recall", "Precisão", "Especificidade"]
    cores = ["#1d4e89", "#b45309", "#2f6f4e", "#6b3fa0"]
    fig, ax = plt.subplots(figsize=(11, 5))
    n_grupos = len(grupos)
    n_metricas = len(nomes_metricas)
    largura = 0.18
    bases = list(range(n_grupos))
    for indice, (campo, rotulo, cor) in enumerate(zip(nomes_metricas, rotulos, cores)):
        deslocamento = (indice - (n_metricas - 1) / 2) * largura
        xs = [base + deslocamento for base in bases]
        ys = [grupo["medias"][campo] for grupo in grupos]
        erros = [grupo["desvios"][campo] for grupo in grupos]
        ax.bar(xs, ys, largura, yerr=erros, label=rotulo, color=cor, capsize=3)
    ax.set_xticks(bases, [grupo["nome"] for grupo in grupos], rotation=12)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Valor no teste")
    ax.set_title("Média das métricas no teste")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    ax.grid(axis="x", visible=False)
    _salvar_figura(fig, "barras_metricas.png")


def _plotar_importancia(itens):
    ordenado = sorted(itens, key=lambda item: item["queda_acuracia_balanceada"])
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(
        [item["variavel"] for item in ordenado],
        [item["queda_acuracia_balanceada"] for item in ordenado],
        color=COR_GOOD,
    )
    ax.axvline(0, color="#666666", linewidth=0.8)
    ax.set_xlabel("Queda da acurácia balanceada ao embaralhar a variável")
    ax.set_title("Importância por permutação no teste")
    _salvar_figura(fig, "importancia_permutacao.png")


def _publicar_execucao(execucao):
    return {
        "semente": execucao["semente"],
        "limiar": execucao["limiar"],
        "metricas_validacao": execucao["metricas_validacao"],
        "metricas_teste": execucao["metricas_teste"],
    }


def _publicar_grupo(grupo):
    return {
        "nome": grupo["nome"],
        "ocultas": grupo["ocultas"],
        "medias": grupo["medias"],
        "desvios": grupo["desvios"],
        "f1_validacao_media": grupo["f1_validacao_media"],
        "execucoes": [_publicar_execucao(execucao) for execucao in grupo["execucoes"]],
    }


def _linha_tabela(nome, medias, desvios):
    linha = {"modelo": nome}
    for campo in CAMPOS:
        linha[f"{campo}_media"] = round(medias[campo], 6)
        linha[f"{campo}_dp"] = round(desvios[campo], 6)
    return linha


def _salvar_tabelas(linhas, importancia):
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    campos = ["modelo"]
    for campo in CAMPOS:
        campos.extend([f"{campo}_media", f"{campo}_dp"])
    with (RESULTADOS / "metricas_teste.csv").open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas)
    with (RESULTADOS / "importancia.csv").open("w", newline="", encoding="utf-8") as arquivo:
        campos_importancia = [
            "variavel",
            "queda_recall",
            "queda_acuracia_balanceada",
            "magnitude_peso",
        ]
        escritor = csv.DictWriter(arquivo, fieldnames=campos_importancia)
        escritor.writeheader()
        for item in importancia:
            escritor.writerow(
                {
                    "variavel": item["variavel"],
                    "queda_recall": round(item["queda_recall"], 6),
                    "queda_acuracia_balanceada": round(item["queda_acuracia_balanceada"], 6),
                    "magnitude_peso": round(item["magnitude_peso"], 6),
                }
            )


def _imprimir_tabela(linhas):
    print("\nResumo no teste (média ± desvio)", flush=True)
    for linha in linhas:
        print(
            f"{linha['modelo']:<22} "
            f"acc {linha['acuracia_media']:.3f}±{linha['acuracia_dp']:.3f}  "
            f"erro {linha['erro_media']:.3f}  "
            f"rec {linha['recall_media']:.3f}±{linha['recall_dp']:.3f}  "
            f"prec {linha['precisao_media']:.3f}±{linha['precisao_dp']:.3f}  "
            f"esp {linha['especificidade_media']:.3f}±{linha['especificidade_dp']:.3f}  "
            f"f1 {linha['f1_media']:.3f}±{linha['f1_dp']:.3f}  "
            f"auc {linha['auc_media']:.3f}±{linha['auc_dp']:.3f}",
            flush=True,
        )


def _json_seguro(objeto):
    if isinstance(objeto, dict):
        return {chave: _json_seguro(valor) for chave, valor in objeto.items()}
    if isinstance(objeto, list):
        return [_json_seguro(valor) for valor in objeto]
    if isinstance(objeto, float) and not math.isfinite(objeto):
        return None
    return objeto


def executar_estudo():
    _estilo()
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)

    erro_gradiente = verificar_gradiente()
    perda_inicial, perda_final = verificar_aprendizado()
    print(
        f"Gradiente conferido (erro máximo {erro_gradiente:.3e}). "
        f"Exemplo de controle: perda {perda_inicial:.3f} -> {perda_final:.3f}.",
        flush=True,
    )

    dados = preparar_dados()
    resumo = dados["resumo"]
    print(
        f"Dataset limpo: {resumo['linhas_finais']} jogadores, "
        f"Good {resumo['good']}, Bad {resumo['bad']}. "
        f"PTS Good [{resumo['pts_good_min']}, {resumo['pts_good_max']}], "
        f"PTS Bad [{resumo['pts_bad_min']}, {resumo['pts_bad_max']}].",
        flush=True,
    )
    print(f"Divisão: {resumo['divisao']}", flush=True)
    _plotar_eda(dados["df_eda"], resumo)

    pesos = pesos_de_classe(dados["y_treino"])
    print(f"Pesos de classe no treino: {pesos}", flush=True)

    pilotos = []
    for taxa in TAXAS:
        print(f"\nPiloto com 16 neurônios e taxa {taxa}", flush=True)
        _, historico = _treinar(
            dados["X_treino"],
            dados["y_treino"],
            dados["X_val"],
            dados["y_val"],
            ocultas=(16,),
            taxa=taxa,
            epocas=EPOCAS_PILOTO,
            semente=1,
            pesos=pesos,
        )
        pilotos.append({"taxa": taxa, "historico": historico})
    _plotar_piloto(pilotos)
    taxa, epocas = escolher_hiperparametros(pilotos)
    print(f"\nHiperparâmetros escolhidos na validação: taxa {taxa}, épocas {epocas}", flush=True)

    grupos = []
    for nome, ocultas in ARQUITETURAS:
        grupos.append(
            _rodar_grupo(
                nome,
                ocultas,
                taxa,
                epocas,
                pesos,
                dados["X_treino"],
                dados["y_treino"],
                dados["X_val"],
                dados["y_val"],
                dados["X_teste"],
                dados["y_teste"],
            )
        )

    melhor_grupo = max(grupos, key=lambda grupo: grupo["f1_validacao_media"])
    melhor_execucao = max(melhor_grupo["execucoes"], key=lambda item: item["metricas_validacao"]["f1"])
    print(
        f"\nArquitetura escolhida pela validação: {melhor_grupo['nome']} "
        f"(F1 médio de validação {melhor_grupo['f1_validacao_media']:.3f}, "
        f"semente {melhor_execucao['semente']}, limiar {melhor_execucao['limiar']:.4f})",
        flush=True,
    )

    grupo_pts = _rodar_grupo(
        "MLP com PTS",
        tuple(melhor_grupo["ocultas"]),
        taxa,
        epocas,
        pesos,
        dados["X_treino_pts"],
        dados["y_treino"],
        dados["X_val_pts"],
        dados["y_val"],
        dados["X_teste_pts"],
        dados["y_teste"],
    )
    melhor_pts = max(grupo_pts["execucoes"], key=lambda item: item["metricas_validacao"]["f1"])
    baseline = _baseline(dados["y_teste"])

    importancia = importancia_por_permutacao(
        melhor_execucao["rede"],
        dados["X_teste"],
        dados["y_teste"],
        melhor_execucao["limiar"],
    )
    magnitudes = melhor_execucao["rede"].magnitude_pesos_entrada()
    for item, nome, magnitude in zip(importancia, dados["nomes"], magnitudes):
        item["variavel"] = nome
        item["magnitude_peso"] = magnitude
    importancia_ordenada = sorted(
        importancia, key=lambda item: item["queda_acuracia_balanceada"], reverse=True
    )

    _plotar_perda(
        melhor_execucao["historico"],
        f"Perda da melhor execução ({melhor_grupo['nome']}, semente {melhor_execucao['semente']})",
    )
    _plotar_matriz(
        melhor_execucao["metricas_teste"],
        f"Teste — {melhor_grupo['nome']}, semente {melhor_execucao['semente']}",
    )
    curvas = []
    for grupo in grupos:
        destaque = max(grupo["execucoes"], key=lambda item: item["metricas_validacao"]["f1"])
        curvas.append(
            {
                "nome": f"{grupo['nome']} (AUC {destaque['metricas_teste']['auc']:.2f})",
                "roc": destaque["roc"],
                "cor": CORES[grupo["nome"]],
            }
        )
    curvas.append(
        {
            "nome": f"MLP com PTS (AUC {melhor_pts['metricas_teste']['auc']:.2f})",
            "roc": melhor_pts["roc"],
            "cor": CORES["MLP com PTS"],
            "estilo": "--",
        }
    )
    _plotar_roc(curvas)

    grupos_barras = [
        {"nome": "Sempre Bad", "medias": baseline, "desvios": {campo: 0.0 for campo in CAMPOS}},
        grupo_pts,
        *grupos,
    ]
    _plotar_barras(grupos_barras)
    _plotar_importancia(importancia_ordenada)

    linhas = [
        _linha_tabela("Sempre Bad", baseline, {campo: 0.0 for campo in CAMPOS}),
        _linha_tabela(grupo_pts["nome"], grupo_pts["medias"], grupo_pts["desvios"]),
    ]
    linhas.extend(_linha_tabela(grupo["nome"], grupo["medias"], grupo["desvios"]) for grupo in grupos)
    _salvar_tabelas(linhas, importancia_ordenada)
    _imprimir_tabela(linhas)

    pacote = {
        "limpeza": resumo,
        "pesos_classe_treino": pesos,
        "piloto": [
            {
                "taxa": piloto["taxa"],
                "perda_validacao_final": piloto["historico"][-1]["perda_validacao"],
                "perda_treino_final": piloto["historico"][-1]["perda_treino"],
                "historico": piloto["historico"],
            }
            for piloto in pilotos
        ],
        "taxa_escolhida": taxa,
        "epocas_escolhidas": epocas,
        "epocas_piloto": EPOCAS_PILOTO,
        "sementes": SEMENTES,
        "baseline": baseline,
        "grupos": [_publicar_grupo(grupo) for grupo in grupos],
        "grupo_pts": _publicar_grupo(grupo_pts),
        "melhor": {
            "arquitetura": melhor_grupo["nome"],
            "ocultas": melhor_grupo["ocultas"],
            "semente": melhor_execucao["semente"],
            "limiar": melhor_execucao["limiar"],
            "f1_validacao_media_da_arquitetura": melhor_grupo["f1_validacao_media"],
            "metricas_validacao": melhor_execucao["metricas_validacao"],
            "metricas_teste": melhor_execucao["metricas_teste"],
            "historico": melhor_execucao["historico"],
        },
        "importancia": importancia_ordenada,
    }
    (RESULTADOS / "resumo.json").write_text(
        json.dumps(_json_seguro(pacote), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nFiguras em {FIGURAS}", flush=True)
    print(f"Tabelas e resumo em {RESULTADOS}", flush=True)
    return pacote
