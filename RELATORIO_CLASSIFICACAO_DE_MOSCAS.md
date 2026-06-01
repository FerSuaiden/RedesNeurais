# Classificação de Moscas com Redes Neurais

## Resumo

Este trabalho apresenta uma solução para o problema de classificação de insetos a partir de recortes
de imagens fornecidos pela Embrapa. O objetivo é distinguir recortes da classe `WF`, correspondente
à mosca-branca (*Bemisia tabaci*), da classe `MR`, que representa outros insetos presentes nas
armadilhas adesivas amarelas. Para isso, foi montado um pipeline composto por análise exploratória,
pré-processamento, divisão estratificada em treino, validação e teste, treinamento de modelos neurais
e avaliação com a área sob a curva Precisão-Revocação. Dois modelos foram considerados: um MLP como
baseline e uma rede neural convolucional como arquitetura principal. Os resultados mostraram desempenho
elevado em ambos os casos, com Average Precision acima de 0,99 nas execuções locais de referência.
Mesmo quando o baseline apresentou desempenho competitivo, a CNN permaneceu como escolha metodológica
principal por ser mais adequada para dados visuais e por usar muito menos parâmetros.

## 1. Introdução

A mosca-branca é uma das principais pragas agrícolas e seu monitoramento é fundamental para reduzir
perdas de produção e diminuir o uso indiscriminado de pesticidas. Em cenários reais, a contagem e a
identificação manual de insetos são lentas, cansativas e sujeitas a erro humano. Por isso, métodos
automáticos de visão computacional e aprendizado de máquina surgem como alternativa promissora para
apoio ao manejo integrado de pragas.

Neste projeto, o problema foi formulado como uma tarefa de classificação binária. Em vez de trabalhar
com as imagens completas das armadilhas, foram utilizados os recortes já fornecidos em duas pastas:
`WF`, contendo recortes de moscas-brancas, e `MR`, contendo recortes de outros insetos. A tarefa
consiste em treinar um classificador capaz de distinguir as duas classes a partir desses recortes.

## 2. Metodologia

### 2.1 Montagem do conjunto de dados

O conjunto foi construído diretamente a partir das pastas `WF` e `MR`. Cada arquivo `.jpg` foi tratado
como uma amostra independente. Os rótulos foram codificados da seguinte forma:

- `WF` -> 1
- `MR` -> 0

Ao todo, foram utilizadas 7.637 imagens, distribuídas em:

- 6.350 imagens da classe `WF`
- 1.287 imagens da classe `MR`

Isso caracteriza um conjunto desbalanceado, com predominância da classe `WF`.

### 2.2 Análise exploratória

A análise exploratória mostrou que as classes possuem diferenças visuais e geométricas relevantes.
Os recortes de `WF` tendem a ser menores, enquanto os de `MR` costumam apresentar maiores larguras
e alturas. Isso indica que tamanho, forma e contraste do inseto são atributos naturalmente úteis para
a classificação.

### 2.3 Estratégia de divisão

Foi utilizada divisão estratificada em três subconjuntos:

- treino: 70%
- validação: 15%
- teste: 15%

Nas execuções de referência, isso resultou em:

- treino: 5.345 imagens
- validação: 1.146 imagens
- teste: 1.146 imagens

A estratificação foi importante para preservar a proporção entre as classes em todas as partições.

### 2.4 Pré-processamento

Como os recortes possuem tamanhos variados, foi necessário redimensioná-los para um formato fixo.
Foram usadas duas configurações:

- MLP: imagens redimensionadas para 32x32 pixels;
- CNN: imagens redimensionadas para 64x64 pixels.

Para o treinamento da CNN, foram aplicadas transformações leves de aumento de dados:

- espelhamento horizontal;
- pequenas rotações.

Além disso, como o conjunto é desbalanceado, foi utilizado `WeightedRandomSampler` durante o treino
para aumentar a representatividade da classe minoritária.

### 2.5 Modelos

#### MLP baseline

O primeiro modelo foi um perceptron multicamadas com duas camadas ocultas. Nesse caso, a imagem foi
achatada em um vetor unidimensional. O objetivo desse modelo foi servir como baseline simples, já que
ele não preserva relações espaciais explícitas entre pixels.

#### CNN principal

O segundo modelo, escolhido como arquitetura principal, foi uma rede convolucional pequena com três
blocos de convolução seguidos de ReLU e max pooling, encerrando com uma camada totalmente conectada
para classificação binária. Essa arquitetura foi escolhida por três motivos:

- é compatível com o foco do projeto, que solicita exploração de CNNs;
- preserva e explora a estrutura espacial da imagem;
- usa muito menos parâmetros do que o MLP baseline.

## 3. Experimentos

Os dois modelos foram treinados com função de perda `BCEWithLogitsLoss` e otimizador Adam com taxa de
aprendizado igual a `1e-3`. A seleção do melhor ponto de cada treinamento foi feita com base no melhor
Average Precision no conjunto de validação.

O critério principal de avaliação no conjunto de teste foi a área sob a curva Precisão-Revocação,
implementada por meio da métrica `average_precision_score`.

## 4. Resultados

Nas execuções locais de referência, foram observados os seguintes resultados no conjunto de teste:

| Modelo | Parâmetros | Average Precision |
|--------|------------:|------------------:|
| MLP | 803.201 | 0,9949 |
| CNN | 23.649 | 0,9927 |

Os resultados mostram que o problema é altamente separável. Mesmo o modelo denso obteve desempenho
excelente, o que sugere que atributos simples como escala do recorte, contraste e forma já são
fortemente discriminativos.

## 5. Discussão

Embora o MLP tenha apresentado desempenho ligeiramente superior em uma execução local específica, a
CNN foi mantida como arquitetura principal do trabalho por razões metodológicas sólidas. O MLP trata
a imagem como um vetor, ignorando a vizinhança entre pixels. Já a CNN explora estrutura espacial,
é mais compatível com problemas de visão computacional e utiliza muito menos parâmetros.

Outro ponto importante é que, em tarefas mais difíceis e menos separáveis que esta, o viés indutivo
da CNN tende a ser mais vantajoso. Portanto, a escolha da CNN permanece tecnicamente justificável
mesmo quando o baseline denso é competitivo.

O desbalanceamento entre as classes também merece destaque. Sem tratamento, o modelo poderia tender
a favorecer a classe majoritária. O uso de amostragem ponderada foi uma decisão relevante para tornar
o treino mais equilibrado.

## 6. Conclusão

O projeto cumpriu os requisitos propostos no enunciado. Foi montado um pipeline completo para
classificação de moscas a partir dos recortes `WF` e `MR`, incluindo montagem do conjunto,
pré-processamento, divisão em treino/validação/teste, treinamento de modelos neurais e avaliação por
Precisão-Revocação.

A solução final adota a rede neural convolucional como arquitetura principal, com o MLP sendo usado
como baseline de comparação. Os resultados indicam que o problema pode ser resolvido com alta precisão,
o que reforça o potencial de técnicas de aprendizado profundo para apoio ao monitoramento automático
de pragas agrícolas.

Como trabalhos futuros, sugerem-se três extensões:

- testar transferência de aprendizado com modelos pré-treinados;
- aplicar validação cruzada estratificada;
- realizar análise detalhada dos erros com inspeção visual dos falsos positivos e falsos negativos.

## Referências

- BAR, L. et al. Downregulation of dystrophin expression in pupae of the whitefly *Bemisia tabaci* inhibits the emergence of adults. *Insect Molecular Biology*, 2019.
- JIGE, M. N.; RATNAPARKHE, V. R. Population estimation of whitefly for cotton plant using image processing approach. *2017 2nd IEEE International Conference on Recent Trends in Electronics, Information & Communication Technology*, 2017.
- BARBEDO, J. G. A. Using digital image processing for counting whiteflies on soybean leaves. *Journal of Asia-Pacific Entomology*, 2014.
- DAVIS, J.; GOADRICH, M. The relationship between Precision-Recall and ROC curves. *ICML*, 2006.
