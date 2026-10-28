"""Rede neural perceptron multicamada, escrita sem biblioteca de machine learning.

O que esta no código, e não em biblioteca:
- somatório de cada neurônio
- função de ativação sigmoide e sua derivada
- entropia cruzada binária com peso de classe
- retropropagação e descida de gradiente em lote

Listas e o módulo math fazem a conta. Nenhuma operação da rede passa por
NumPy, scikit-learn ou framework de redes neurais.
"""

import math
import random


def sigmoide(z):
    """σ(z) = 1 / (1 + e^(-z)).

    O corte em ±500 evita que math.exp estoure. Fora desse intervalo a
    sigmoide já vale, na prática, 0 ou 1.
    """
    if z >= 0.0:
        if z > 500.0:
            return 1.0
        expoente = math.exp(-z)
        return 1.0 / (1.0 + expoente)
    if z < -500.0:
        return 0.0
    expoente = math.exp(z)
    return expoente / (1.0 + expoente)


def entropia_cruzada_binaria(previsto, real, peso):
    """Perda de uma amostra.

    L = -peso * [y * ln(p) + (1 - y) * ln(1 - p)]
    """
    probabilidade = min(max(previsto, 1e-12), 1.0 - 1e-12)
    return -peso * (
        real * math.log(probabilidade)
        + (1.0 - real) * math.log(1.0 - probabilidade)
    )


def pesos_de_classe(rotulos):
    """Peso balanceado: n / (2 * quantidade da classe).

    Sem esse peso, a classe Bad (maioria) domina a perda e a rede aprende
    a responder sempre Bad.
    """
    n = len(rotulos)
    n_good = sum(int(rotulo) for rotulo in rotulos)
    n_bad = n - n_good
    if n_good == 0 or n_bad == 0:
        raise ValueError("O conjunto de treino precisa ter as duas classes.")
    return {0: n / (2.0 * n_bad), 1: n / (2.0 * n_good)}


class RedeMLP:
    """MLP com camadas ocultas sigmoides e um neurônio de saída sigmoide."""

    def __init__(self, tamanhos, semente, pesos_classe):
        if len(tamanhos) < 2:
            raise ValueError("A rede precisa de uma entrada e uma saída.")
        self.tamanhos = list(tamanhos)
        self.pesos_classe = pesos_classe
        self.rng = random.Random(semente)
        self.pesos = []
        self.biases = []
        for fan_in, fan_out in zip(self.tamanhos[:-1], self.tamanhos[1:]):
            # Xavier uniforme, adequado à sigmoide: U(-L, L), L = sqrt(6 / (entrada + saída)).
            limite = math.sqrt(6.0 / (fan_in + fan_out))
            camada = []
            for _ in range(fan_out):
                camada.append([self.rng.uniform(-limite, limite) for _ in range(fan_in)])
            self.pesos.append(camada)
            self.biases.append([0.0 for _ in range(fan_out)])
        self.grad_w = []
        self.grad_b = []
        for camada in self.pesos:
            self.grad_w.append([[0.0] * len(neuronio) for neuronio in camada])
            self.grad_b.append([0.0] * len(camada))

    def propagar(self, entradas):
        """Calcula a ativação de cada camada. A primeira lista é a entrada."""
        ativacoes = [entradas]
        sinal = entradas
        for pesos, biases in zip(self.pesos, self.biases):
            saida = []
            for pesos_neuronio, bias in zip(pesos, biases):
                acumulado = bias
                for valor, peso in zip(sinal, pesos_neuronio):
                    acumulado += valor * peso
                saida.append(sigmoide(acumulado))
            ativacoes.append(saida)
            sinal = saida
        return ativacoes

    def prever_proba(self, amostras):
        """Probabilidade de Good (classe 1) segundo a sigmoide de saída."""
        return [self.propagar(amostra)[-1][0] for amostra in amostras]

    def prever(self, amostras, limiar):
        return [1 if probabilidade >= limiar else 0 for probabilidade in self.prever_proba(amostras)]

    def _limpar_gradientes(self):
        for camada in self.grad_w:
            for neuronio in camada:
                for indice in range(len(neuronio)):
                    neuronio[indice] = 0.0
        for camada in self.grad_b:
            for indice in range(len(camada)):
                camada[indice] = 0.0

    def _acumular_gradiente(self, entradas, real):
        """Retropropagação de uma amostra. Soma o gradiente nos acumuladores.

        Saída, com sigmoide e entropia cruzada:
            delta = peso_da_classe * (previsto - real)
        Camada oculta:
            delta_j = (soma_k peso_kj * delta_k) * a_j * (1 - a_j)
        em que a_j * (1 - a_j) é a derivada da sigmoide.
        Gradiente do peso que liga i a j:
            dL/dW_ji = delta_j * ativação_i
        """
        pesos = self.pesos
        biases = self.biases
        n_camadas = len(pesos)
        ativacoes = [entradas]
        sinal = entradas
        for indice_camada in range(n_camadas):
            pesos_camada = pesos[indice_camada]
            biases_camada = biases[indice_camada]
            n_neuronios = len(pesos_camada)
            n_entradas = len(sinal)
            saida = [0.0] * n_neuronios
            for j in range(n_neuronios):
                pesos_neuronio = pesos_camada[j]
                acumulado = biases_camada[j]
                for i in range(n_entradas):
                    acumulado += sinal[i] * pesos_neuronio[i]
                saida[j] = sigmoide(acumulado)
            ativacoes.append(saida)
            sinal = saida

        previsto = ativacoes[-1][0]
        peso_classe = self.pesos_classe[int(real)]
        perda = entropia_cruzada_binaria(previsto, real, peso_classe)
        delta = [peso_classe * (previsto - real)]

        for indice_camada in range(n_camadas - 1, -1, -1):
            entrada_camada = ativacoes[indice_camada]
            pesos_camada = pesos[indice_camada]
            gradiente_pesos = self.grad_w[indice_camada]
            gradiente_bias = self.grad_b[indice_camada]
            n_entradas = len(entrada_camada)
            for j, delta_j in enumerate(delta):
                gradiente_bias[j] += delta_j
                pesos_neuronio = pesos_camada[j]
                gradiente_neuronio = gradiente_pesos[j]
                for i in range(n_entradas):
                    gradiente_neuronio[i] += delta_j * entrada_camada[i]
            if indice_camada == 0:
                break
            delta_oculto = [0.0] * n_entradas
            for i in range(n_entradas):
                soma = 0.0
                for j, delta_j in enumerate(delta):
                    soma += pesos_camada[j][i] * delta_j
                ativacao = entrada_camada[i]
                delta_oculto[i] = soma * ativacao * (1.0 - ativacao)
            delta = delta_oculto
        return perda

    def perda_media(self, amostras, rotulos):
        if not amostras:
            return 0.0
        total = 0.0
        for entradas, real in zip(amostras, rotulos):
            previsto = self.propagar(entradas)[-1][0]
            total += entropia_cruzada_binaria(previsto, real, self.pesos_classe[int(real)])
        return total / len(amostras)

    def treinar(self, amostras, rotulos, amostras_val, rotulos_val, taxa, epocas, a_cada=50):
        """Descida de gradiente em lote: a média do gradiente de todas as amostras."""
        n = len(amostras)
        if n == 0:
            raise ValueError("O treino está vazio.")
        historico = []
        for epoca in range(1, epocas + 1):
            self._limpar_gradientes()
            perda = 0.0
            for entradas, real in zip(amostras, rotulos):
                perda += self._acumular_gradiente(entradas, real)
            perda /= n
            passo = taxa / n
            for pesos, biases, gradiente_pesos, gradiente_bias in zip(
                self.pesos, self.biases, self.grad_w, self.grad_b
            ):
                for j in range(len(biases)):
                    biases[j] -= passo * gradiente_bias[j]
                    pesos_neuronio = pesos[j]
                    gradiente_neuronio = gradiente_pesos[j]
                    for i in range(len(pesos_neuronio)):
                        pesos_neuronio[i] -= passo * gradiente_neuronio[i]
            perda_val = self.perda_media(amostras_val, rotulos_val)
            if not math.isfinite(perda) or not math.isfinite(perda_val):
                raise RuntimeError(f"A perda deixou de ser finita na época {epoca}.")
            historico.append(
                {"epoca": epoca, "perda_treino": perda, "perda_validacao": perda_val}
            )
            if a_cada and epoca % a_cada == 0:
                print(
                    f"    época {epoca:4d}  treino {perda:.4f}  validação {perda_val:.4f}",
                    flush=True,
                )
        return historico

    def magnitude_pesos_entrada(self):
        """Média do módulo dos pesos que saem de cada variável de entrada."""
        primeira = self.pesos[0]
        n_entradas = len(primeira[0])
        magnitudes = []
        for i in range(n_entradas):
            soma = 0.0
            for neuronio in primeira:
                soma += abs(neuronio[i])
            magnitudes.append(soma / len(primeira))
        return magnitudes


def verificar_gradiente(tolerancia=1e-4):
    """Confere a retropropagação com diferenças finitas em uma rede pequena."""
    tamanhos = [4, 3, 2, 1]
    pesos_classe = {0: 1.0, 1: 2.0}
    rede = RedeMLP(tamanhos, semente=7, pesos_classe=pesos_classe)
    gerador = random.Random(0)
    entradas = [gerador.uniform(-1.0, 1.0) for _ in range(4)]
    real = 1.0
    rede._limpar_gradientes()
    rede._acumular_gradiente(entradas, real)
    epsilon = 1e-5
    pontos = [(0, 1, 2), (1, 0, 1), (2, 0, 1)]
    maior_erro = 0.0
    for camada, neuronio, peso in pontos:
        analitico = rede.grad_w[camada][neuronio][peso]
        original = rede.pesos[camada][neuronio][peso]
        rede.pesos[camada][neuronio][peso] = original + epsilon
        perda_mais = entropia_cruzada_binaria(
            rede.propagar(entradas)[-1][0], real, pesos_classe[1]
        )
        rede.pesos[camada][neuronio][peso] = original - epsilon
        perda_menos = entropia_cruzada_binaria(
            rede.propagar(entradas)[-1][0], real, pesos_classe[1]
        )
        rede.pesos[camada][neuronio][peso] = original
        numerico = (perda_mais - perda_menos) / (2.0 * epsilon)
        maior_erro = max(maior_erro, abs(analitico - numerico))

    analitico_bias = rede.grad_b[0][0]
    original_bias = rede.biases[0][0]
    rede.biases[0][0] = original_bias + epsilon
    perda_mais = entropia_cruzada_binaria(
        rede.propagar(entradas)[-1][0], real, pesos_classe[1]
    )
    rede.biases[0][0] = original_bias - epsilon
    perda_menos = entropia_cruzada_binaria(
        rede.propagar(entradas)[-1][0], real, pesos_classe[1]
    )
    rede.biases[0][0] = original_bias
    numerico_bias = (perda_mais - perda_menos) / (2.0 * epsilon)
    maior_erro = max(maior_erro, abs(analitico_bias - numerico_bias))
    if maior_erro > tolerancia:
        raise RuntimeError(f"Gradiente divergente da diferença finita: erro {maior_erro}")
    return maior_erro


def verificar_aprendizado():
    """A perda precisa cair num problema pequeno e separável."""
    gerador = random.Random(1)
    amostras = []
    rotulos = []
    for _ in range(80):
        x1 = gerador.uniform(-1.0, 1.0)
        x2 = gerador.uniform(-1.0, 1.0)
        amostras.append([x1, x2])
        rotulos.append(1 if x1 + x2 > 0.0 else 0)
    rede = RedeMLP([2, 4, 1], semente=1, pesos_classe=pesos_de_classe(rotulos))
    historico = rede.treinar(amostras, rotulos, amostras, rotulos, taxa=0.5, epocas=60, a_cada=0)
    if historico[-1]["perda_treino"] >= historico[0]["perda_treino"]:
        raise RuntimeError("A perda do exemplo de controle não diminuiu.")
    return historico[0]["perda_treino"], historico[-1]["perda_treino"]
