# Classificação de Moscas com Redes Neurais

## Resumo

Este trabalho investiga a classificação binária de recortes de insetos fornecidos pela Embrapa,
distinguindo amostras da classe `WF`, correspondente à mosca-branca, de amostras da classe `MR`,
que reúnem outros insetos presentes nas armadilhas. Foram avaliadas três abordagens: um baseline
geométrico com atributos simples do bounding box, um perceptron multicamadas (MLP) e uma rede
neural convolucional (CNN). Durante a revisão metodológica, observou-se que uma divisão ingênua por
recorte gera vazamento entre treino e teste, pois vários recortes pertencem à mesma imagem original.
Por isso, a avaliação final foi refeita com split por imagem de origem. Os resultados mostram que o
dataset é altamente separável por geometria, já que um classificador logístico com largura, altura,
área e razão de aspecto alcança Average Precision próxima de 0,9994 no teste. Ainda assim, a CNN
apresentou o melhor equilíbrio entre Average Precision e comportamento operacional no threshold 0,5,
superando o MLP em precisão para a classe minoritária `MR`.

## 1. Introdução

A mosca-branca é uma praga de grande relevância agrícola, e seu monitoramento é importante tanto
para controle químico quanto para estratégias de manejo integrado. O uso de imagens para
identificação automática de insetos pode reduzir esforço manual e tornar o processo mais
consistente.

Neste projeto, o problema foi formulado como classificação binária a partir de recortes já
extraídos das armadilhas adesivas. As amostras estão organizadas em duas pastas, `WF` e `MR`. O
objetivo foi avaliar arquiteturas de redes neurais e, ao mesmo tempo, analisar com cuidado quais
pistas do dataset estão realmente sendo exploradas pelos modelos.

## 2. Metodologia

### 2.1 Montagem do conjunto de dados

Cada arquivo `.jpg` das pastas `WF` e `MR` foi tratado como uma amostra. Além do rótulo binário,
também foi extraído o identificador da imagem original a partir do nome do arquivo. Assim, arquivos
como `1000_WF_001.jpg` e `1000_MR_001.jpg` são reconhecidos como recortes derivados da mesma imagem
original `1000`.

Também foram calculados atributos geométricos simples: largura do recorte, altura, área e razão de
aspecto. Esses atributos foram incluídos porque a análise exploratória mostrou que `WF` tende a ser
bem menor do que `MR`, o que sugere forte poder discriminativo da geometria do bounding box.

### 2.2 Análise exploratória

O conjunto possui forte desbalanceamento, com 6.350 amostras `WF` e 1.287 amostras `MR`. Além
disso, os recortes de `WF` são, em média, menores e mais compactos, enquanto os de `MR` tendem a
ser maiores e mais alongados. Essa diferença é importante porque indica a presença de um atalho
geométrico forte no dataset. Em outras palavras, parte relevante do problema pode ser resolvida sem
que o modelo aprenda características visuais complexas.

### 2.3 Divisão treino/validação/teste sem vazamento

Uma primeira versão do pipeline dividia o conjunto diretamente por recorte. Essa abordagem é
problemática porque milhares de recortes foram gerados a partir de apenas 284 imagens originais. Em
termos práticos, isso significa que o modelo poderia treinar com alguns recortes da imagem `1000`
e depois ser avaliado com outros recortes dessa mesma imagem. Nessa situação, o teste deixa de
representar uma fotografia realmente nova e se torna artificialmente mais fácil.

Para corrigir esse problema, a divisão final foi feita por grupo, usando o identificador da imagem
original como chave. Foi utilizado `GroupShuffleSplit` em duas etapas para obter partições próximas
de 70% para treino, 15% para validação e 15% para teste. Na execução local de referência, isso
resultou em 5.569 amostras distribuídas em 148 grupos no treino, 936 amostras em 32 grupos na
validação e 1.132 amostras também em 32 grupos no teste. Com esse procedimento, não há
sobreposição de imagens originais entre treino, validação e teste.

### 2.4 Pré-processamento

Para o MLP, as imagens foram redimensionadas para 32x32 pixels. Para a CNN, adotou-se um
pré-processamento mais cuidadoso, composto por padding para transformar o recorte em formato
quadrado, redimensionamento para 64x64, espelhamento horizontal aleatório, pequenas rotações e
normalização dos canais.

O padding teve papel importante, porque os recortes possuem proporções muito diferentes. Sem esse
passo, o redimensionamento direto deforma o inseto e altera artificialmente sua geometria. Como o
conjunto é desbalanceado, também foi utilizado `WeightedRandomSampler` durante o treino dos modelos
neurais.

### 2.5 Modelos avaliados

O primeiro experimento foi um baseline geométrico, treinado com regressão logística sobre largura,
altura, área e razão de aspecto. Seu papel foi medir quanto do desempenho pode ser explicado apenas
pela geometria do bounding box.

O segundo modelo foi um MLP. Nesse caso, a imagem é achatada em um vetor unidimensional, o que
elimina a organização local entre pixels vizinhos. O MLP foi mantido como baseline neural porque
permite comparação direta com a CNN sem um mecanismo explícito para explorar contornos, texturas e
formas locais.

O terceiro modelo foi a CNN principal. A arquitetura adotada usa blocos convolucionais com
`BatchNorm`, `ReLU`, `MaxPool`, `AdaptiveAvgPool` e `Dropout`. Essa escolha se justifica porque a
entrada do problema é visual. Em uma imagem, pixels vizinhos formam padrões locais, como regiões
claras, bordas, alongamentos e contornos. A convolução foi concebida precisamente para explorar
esse tipo de organização.

## 3. Experimentos

Os modelos neurais foram treinados com `BCEWithLogitsLoss` e otimizador Adam. O MLP utilizou taxa
de aprendizado `1e-3`, enquanto a CNN utilizou `8e-4`. A seleção do melhor ponto foi feita com
base no Average Precision no conjunto de validação.

Além do Average Precision, também foram observadas métricas no threshold de decisão 0,5, com
atenção especial à classe `MR`, por ser a classe minoritária e por representar um cenário
operacional mais sensível.

Convém distinguir duas quantidades presentes nos resultados. A `loss` é a função numérica usada no
treinamento; valores menores indicam melhor ajuste do modelo aos rótulos observados. A `Average
Precision`, abreviada como `AP`, resume a curva Precisão-Revocação considerando todos os limiares
possíveis de decisão. Portanto, `loss` e `AP` descrevem aspectos diferentes do comportamento do
modelo.

## 4. Resultados

Na execução local de referência com split por grupo, foram obtidos os seguintes resultados no
conjunto de teste:

| Modelo | Parâmetros | Average Precision | Precisão MR @0,5 | Revocação MR @0,5 | Acurácia @0,5 |
|--------|------------:|------------------:|-----------------:|------------------:|--------------:|
| Geométrico | não se aplica | 0,9994 | 0,9429 | 0,9593 | 0,9850 |
| MLP | 803.201 | 0,9963 | 0,6457 | 0,9535 | 0,9134 |
| CNN | 79.225 | 0,99999 | 0,9829 | 1,0000 | 0,9974 |

Esses resultados sustentam três observações centrais. Primeiro, o dataset é quase totalmente
separável por geometria simples. Segundo, AP sozinha não basta para descrever o comportamento do
modelo. Terceiro, a CNN foi a melhor entre as abordagens neurais tanto em AP quanto em precisão
para `MR`.

O valor `1,0000` presente na coluna de revocação de `MR` não significa que a CNN acerta tudo. Esse
número indica apenas que, nesse teste e nesse threshold, todas as amostras reais de `MR` foram
recuperadas. Isso não implica precisão perfeita nem ausência de erros. A própria precisão de `MR`
ficou em `0,9829`, o que mostra a existência de falsos positivos.

## 5. Discussão

O principal ajuste metodológico deste trabalho foi eliminar o vazamento entre treino e teste. A
divisão por recorte superestimava a independência entre as partições, porque recortes da mesma
imagem original compartilhavam contexto visual e poderiam cair em conjuntos diferentes.

Outro ponto central é que o dataset apresenta forte viés geométrico. O baseline com apenas quatro
atributos simples já alcançou AP muito próxima de 1. Isso significa que uma parte considerável do
desempenho observado não deve ser interpretada automaticamente como evidência de aprendizado visual
complexo.

Também foi necessário relativizar o uso isolado do Average Precision. O MLP, por exemplo, obteve AP
alta, mas teve precisão bastante inferior para `MR` no threshold 0,5. Em um cenário prático, isso
significa mais falsos positivos nessa classe, apesar da boa ordenação global das probabilidades.

Dentro desse contexto, a CNN ainda se mostrou a escolha mais forte para a apresentação final. Ela é
mais apropriada para imagens, superou o MLP em AP e em métricas operacionais, manteve desempenho
muito alto mesmo sob avaliação sem vazamento e permanece alinhada com a proposta de explorar redes
convolucionais.

## 6. Conclusão

O projeto atendeu ao enunciado com uma avaliação mais rigorosa e tecnicamente mais correta. Os
dados foram montados a partir das pastas `WF` e `MR`, o split foi corrigido para respeitar a imagem
original, foram treinados modelos neurais e a avaliação principal foi feita com
Precisão-Revocação.

Os resultados mostram que o problema é altamente separável, que a geometria do recorte é uma pista
extremamente forte, que a CNN é a melhor escolha final entre os modelos neurais avaliados e que a
interpretação dos resultados precisa reconhecer explicitamente as limitações do dataset.

## Referências

BAR, L. et al. Downregulation of dystrophin expression in pupae of the whitefly *Bemisia tabaci*
inhibits the emergence of adults. *Insect Molecular Biology*, 2019.

JIGE, M. N.; RATNAPARKHE, V. R. Population estimation of whitefly for cotton plant using image
processing approach. *2017 2nd IEEE International Conference on Recent Trends in Electronics,
Information & Communication Technology*, 2017.

BARBEDO, J. G. A. Using digital image processing for counting whiteflies on soybean leaves.
*Journal of Asia-Pacific Entomology*, 2014.

DAVIS, J.; GOADRICH, M. The relationship between Precision-Recall and ROC curves. *ICML*, 2006.
