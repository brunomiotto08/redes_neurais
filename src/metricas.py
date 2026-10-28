"""Métricas calculadas a partir da matriz de confusão.

A biblioteca entra só para montar a matriz (confusion_matrix).
Acurácia, erro, recall, precisão, especificidade, F1, ROC e AUC
são contas deste arquivo.
"""

import random

from sklearn.metrics import confusion_matrix


def matriz_confusao(y_real, y_previsto):
    """Ordem fixa dos rótulos [0, 1]: [[TN, FP], [FN, TP]]."""
    matriz = confusion_matrix(y_real, y_previsto, labels=[0, 1])
    return {
        "tn": int(matriz[0][0]),
        "fp": int(matriz[0][1]),
        "fn": int(matriz[1][0]),
        "tp": int(matriz[1][1]),
    }


def calcular_metricas(y_real, y_previsto):
    """Métricas do enunciado. A classe positiva é Good (1).

    Acurácia = (VP + VN) / total
    Erro = 1 - acurácia
    Recall (sensitividade) = VP / (VP + FN)
    Precisão = VP / (VP + FP)
    Especificidade = VN / (VN + FP)
    """
    contagem = matriz_confusao(y_real, y_previsto)
    vp = contagem["tp"]
    vn = contagem["tn"]
    fp = contagem["fp"]
    fn = contagem["fn"]
    total = vp + vn + fp + fn
    acuracia = (vp + vn) / total if total else 0.0
    recall = vp / (vp + fn) if (vp + fn) else 0.0
    precisao = vp / (vp + fp) if (vp + fp) else 0.0
    especificidade = vn / (vn + fp) if (vn + fp) else 0.0
    if precisao + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precisao * recall / (precisao + recall)
    return {
        "acuracia": acuracia,
        "erro": 1.0 - acuracia,
        "recall": recall,
        "precisao": precisao,
        "especificidade": especificidade,
        "f1": f1,
        "acuracia_balanceada": 0.5 * (recall + especificidade),
        "precisao_definida": (vp + fp) > 0,
        **contagem,
    }


def _ponto_roc(y_real, probabilidades, limiar):
    previsto = [1 if probabilidade >= limiar else 0 for probabilidade in probabilidades]
    metricas = calcular_metricas(y_real, previsto)
    return {
        "limiar": limiar,
        "fpr": 1.0 - metricas["especificidade"],
        "tpr": metricas["recall"],
        "f1": metricas["f1"],
        "metricas": metricas,
    }


def curva_roc(y_real, probabilidades):
    """Varre os limiares da própria probabilidade. Sem roc_curve de biblioteca.

    Começa com todos classificados como Bad (TPR = 0, FPR = 0) e termina
    com todos classificados como Good (TPR = 1, FPR = 1).
    """
    if not probabilidades:
        return [
            {"limiar": None, "fpr": 0.0, "tpr": 0.0},
            {"limiar": None, "fpr": 1.0, "tpr": 1.0},
        ]
    pontos = [{"limiar": None, "fpr": 0.0, "tpr": 0.0}]
    for limiar in sorted(set(probabilidades), reverse=True):
        ponto = _ponto_roc(y_real, probabilidades, limiar)
        pontos.append({"limiar": ponto["limiar"], "fpr": ponto["fpr"], "tpr": ponto["tpr"]})
    pontos.append({"limiar": None, "fpr": 1.0, "tpr": 1.0})
    return pontos


def area_sob_roc(pontos):
    """AUC pela regra do trapézio ao longo da curva ROC."""
    area = 0.0
    for indice in range(1, len(pontos)):
        dx = pontos[indice]["fpr"] - pontos[indice - 1]["fpr"]
        altura_media = (pontos[indice]["tpr"] + pontos[indice - 1]["tpr"]) / 2.0
        area += dx * altura_media
    if area < 0.0:
        return 0.0
    if area > 1.0:
        return 1.0
    return area


def melhor_limiar(y_real, probabilidades):
    """Escolhe o limiar que maximiza o F1 de Good. Empate fica com o menor limiar."""
    if not probabilidades:
        raise ValueError("Não há probabilidades para escolher o limiar.")
    melhor = None
    for limiar in sorted(set(probabilidades)):
        ponto = _ponto_roc(y_real, probabilidades, limiar)
        if melhor is None or ponto["f1"] > melhor["f1"]:
            melhor = ponto
    return melhor


def importancia_por_permutacao(rede, amostras, rotulos, limiar, repeticoes=10, semente=123):
    """Queda de desempenho ao embaralhar uma variável de cada vez no teste."""
    base = calcular_metricas(rotulos, rede.prever(amostras, limiar))
    gerador = random.Random(semente)
    n_features = len(amostras[0])
    resultado = []
    for coluna in range(n_features):
        quedas_recall = []
        quedas_balanceada = []
        for _ in range(repeticoes):
            permutada = [list(linha) for linha in amostras]
            valores = [linha[coluna] for linha in permutada]
            gerador.shuffle(valores)
            for linha, valor in zip(permutada, valores):
                linha[coluna] = valor
            metricas = calcular_metricas(rotulos, rede.prever(permutada, limiar))
            quedas_recall.append(base["recall"] - metricas["recall"])
            quedas_balanceada.append(base["acuracia_balanceada"] - metricas["acuracia_balanceada"])
        resultado.append(
            {
                "indice": coluna,
                "queda_recall": sum(quedas_recall) / len(quedas_recall),
                "queda_acuracia_balanceada": sum(quedas_balanceada) / len(quedas_balanceada),
            }
        )
    return resultado
