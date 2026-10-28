# Roteiro da apresentação

Modelo da professora: 7 slides. Cole o texto de cada bloco no slide correspondente e as figuras indicadas. Os trechos “O que falar” ficam na sua fala, não no slide.

Preencha o slide 1 com os nomes da dupla e o semestre. O relatório completo está em `relatorio/RELATORIO.md`.

## Slide 1 — Capa

Análise dos Resultados

Disciplina: FBX4054 — Inteligência Computacional

Alunos: preencher

Semestre: preencher

Trabalho: MLP do zero para classificar jogadores da NBA em Good ou Bad

O que falar: o objetivo foi implementar a rede sem biblioteca de machine learning e avaliar com acurácia, erro, recall, precisão, especificidade e ROC. A classe positiva é Good.

## Slide 2 — Comparação entre resultados

O que foi comparado, sempre no mesmo teste (102 jogadores, 10 Good):

- Baseline: prever sempre Bad
- MLP com PTS, só como diagnóstico
- Três arquiteturas sem volume de cesta: 8 neurônios; 16 neurônios; 16 e 8 neurônios

Protocolo:

- 5 execuções por arquitetura, mudando só a semente dos pesos
- Taxa 0,1 e 300 épocas, escolhidas na validação
- Limiar escolhido na validação pelo melhor F1 de Good
- Arquitetura oficial escolhida pelo F1 médio de validação

Para cada execução: acurácia, erro, recall, especificidade, precisão, F1 e AUC da ROC.

O que falar: a divisão treino/validação/teste é fixa (304 / 102 / 102) e estratificada. A validação não entra no teste. Cinco sementes mostram que o resultado não depende de um sorteio feliz dos pesos.

## Slide 3 — Acurácia e erro

Acurácia = (VP + VN) / (VP + FP + VN + FN)

Erro = 1 − acurácia

No teste, prever sempre Bad já tem acurácia 0,902 e erro 0,098, com recall 0.

Rede oficial (16 neurônios, média de 5 execuções): acurácia 0,965 ± 0,009 e erro 0,035.

Execução usada na matriz (semente 4): VP = 10, FN = 0, FP = 3, VN = 89.

Acurácia = (10 + 89) / 102 = 0,971. Erro = 0,029.

Figura: `figuras/matriz_confusao.png`

O que falar: acurácia alta sozinha engana, porque 90% dos jogadores do teste são Bad. O baseline acerta 0,902 sem encontrar nenhum Good. A rede sobe a acurácia e, principalmente, passa a encontrar a classe rara.

## Slide 4 — Recall, precisão e especificidade

Recall = VP / (VP + FN). Porcentagem dos Good reais que a rede acerta.

Precisão = VP / (VP + FP). Porcentagem dos previstos como Good que realmente são Good.

Especificidade = VN / (VN + FP). Porcentagem dos Bad reais identificados como Bad.

Médias no teste:

| Modelo | Recall | Precisão | Especificidade |
| --- | ---: | ---: | ---: |
| Sempre Bad | 0,000 | 0,000 | 1,000 |
| MLP com PTS | 1,000 ± 0,000 | 0,821 ± 0,029 | 0,976 ± 0,005 |
| 8 neurônios | 0,800 ± 0,200 | 0,772 ± 0,069 | 0,974 ± 0,010 |
| 16 neurônios | 0,880 ± 0,110 | 0,787 ± 0,030 | 0,974 ± 0,006 |
| 16 e 8 neurônios | 0,880 ± 0,130 | 0,698 ± 0,125 | 0,954 ± 0,031 |

Na matriz da semente 4: recall = 10/10 = 1; precisão = 10/13 = 0,769; especificidade = 89/92 = 0,967.

Figura: `figuras/barras_metricas.png`

O que falar: a precisão do baseline é indefinida, porque ele não prevê ninguém como Good; na tabela o valor está 0. A rede de 16 neurônios mantém especificidade 0,974 e sobe o recall de 0 para 0,880. O desvio de 0,110 existe porque o teste só tem 10 Good: cada jogador vale 0,10 de recall. Duas camadas ocultas tiveram a pior precisão e ainda estavam aprendendo na época 300, porque a sigmoide empilhada deixa o gradiente pequeno.

## Slide 5 — Curva ROC

TPR = VP / (VP + FN). É o recall.

FPR = FP / (VN + FP). Porcentagem dos Bad classificados como Good.

Valor ideal: TPR = 1 e FPR = 0.

AUC média no teste:

- Sempre Bad: 0,500
- Três arquiteturas sem pontos: 0,982 a 0,990
- Rede oficial, 16 neurônios: 0,990 ± 0,003
- MLP com PTS: 0,999 ± 0,001

Figura: `figuras/roc_arquiteturas.png`

O que falar: a diagonal é o classificador aleatório. As curvas da rede sobem quase na vertical, perto do canto ideal. Com PTS a AUC chega a 0,999 porque a classe é quase um limiar de pontos. Sem PTS a AUC continua em 0,990: minutos, assistências e turnovers ainda separam os pontuadores.

## Slide 6 — Resultados

Média ± desvio de 5 execuções no teste.

| Modelo | Acurácia | Erro | Recall | Precisão | Especificidade | F1 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sempre Bad | 0,902 | 0,098 | 0,000 | 0,000 | 1,000 | 0,000 | 0,500 |
| MLP com PTS | 0,978 ± 0,004 | 0,022 | 1,000 | 0,821 ± 0,029 | 0,976 ± 0,005 | 0,901 ± 0,018 | 0,999 ± 0,001 |
| 8 neurônios | 0,957 ± 0,016 | 0,043 | 0,800 ± 0,200 | 0,772 ± 0,069 | 0,974 ± 0,010 | 0,775 ± 0,105 | 0,988 ± 0,004 |
| 16 neurônios | 0,965 ± 0,009 | 0,035 | 0,880 ± 0,110 | 0,787 ± 0,030 | 0,974 ± 0,006 | 0,828 ± 0,054 | 0,990 ± 0,003 |
| 16 e 8 neurônios | 0,947 ± 0,026 | 0,053 | 0,880 ± 0,130 | 0,698 ± 0,125 | 0,954 ± 0,031 | 0,770 ± 0,097 | 0,982 ± 0,006 |

Configuração escolhida: 16 neurônios, taxa 0,1, 300 épocas. Critério: maior F1 médio na validação (0,821, contra 0,808 e 0,765).

O que falar: com PTS a rede quase copia a etiqueta. Sem as variáveis de cesta, 16 neurônios é a melhor configuração estável. Oito neurônios fica perto e varia mais. Dezesseis e oito fica atrás neste orçamento de épocas. A perda da execução escolhida caiu de 0,67 para 0,22 na validação, junto com o treino. Figura de apoio, se couber: `figuras/curva_perda_melhor.png` e `figuras/curva_perda_piloto.png`.

## Slide 7 — Conclusões finais

Conclusão: a MLP do zero classifica Good e Bad. A etiqueta Good é alto volume de pontos (todo Good tem pelo menos 20,5 pontos; o único Bad acima de 20 é Terry Rozier, com 21,4). Sem ver os pontos, a rede ainda encontra esses jogadores porque reconhece quem joga muito e participa das posses.

O que mais pesa, por permutação no teste: TOV, AST, MP e GS. Percentuais de arremesso quase não mudam o resultado. FG% médio é 0,484 nos Good e 0,453 nos Bad.

Figura: `figuras/importancia_permutacao.png`

Aprendizados:

- Acurácia de 90% pode ser só a classe majoritária.
- Peso de classe e limiar na validação (aqui entre 0,830 e 0,862) são necessários quando há poucos Good.
- Somatório, sigmoide, entropia cruzada e retropropagação foram implementados e o gradiente foi conferido por diferenças finitas.

Trabalhos futuros: mais temporadas; validação cruzada, porque o teste tem só 10 Good; mais épocas para a rede de duas camadas; repetir a análise depois de retirar minutos, titularidade, assistências e turnovers, para ver se resta sinal de eficiência.

O que falar: turnover na frente não quer dizer que errar passe seja bom. Quem pontua segura mais a bola e, por isso, também perde mais a bola. A rede aprendeu o papel do jogador, que neste dataset é quase a mesma coisa que a etiqueta Good. Se a pergunta fosse eficiência, o dataset quase não separa as classes por FG%.
