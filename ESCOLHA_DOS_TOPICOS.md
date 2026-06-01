# Escolha dos Tópicos da Disciplina

## Tópico principal: Redes Neurais Convolucionais

Este foi o tópico principal escolhido porque o projeto envolve classificação de imagens. Em imagens,
os pixels vizinhos possuem relações espaciais importantes, e as CNNs exploram exatamente esse tipo
de estrutura usando filtros convolucionais. Isso permite aprender bordas, formas e texturas com
muito menos parâmetros do que uma rede totalmente conectada.

Justificativas:

- o enunciado pede explicitamente a exploração de arquiteturas convolucionais;
- a entrada é visual e contém padrões locais relevantes;
- CNNs são o padrão mais adequado para tarefas de visão computacional;
- o modelo convolucional usado no projeto tem menos parâmetros do que o MLP baseline e ainda assim
  mantém desempenho muito alto.

## Tópico secundário: Perceptron / MLP

O MLP foi incluído como baseline comparativo. A ideia não é adotá-lo como arquitetura principal,
mas mostrar experimentalmente a diferença entre uma rede densa sobre pixels achatados e uma rede
que respeita a estrutura espacial da imagem.

Justificativas:

- é um tópico central da disciplina;
- serve como referência simples e interpretável;
- fortalece a justificativa da CNN ao permitir comparação quantitativa.

## Tópico considerado, mas não adotado como principal: Transferência de Aprendizado

Transferência de aprendizado seria uma escolha válida, especialmente com arquiteturas como ResNet
ou MobileNet pré-treinadas em ImageNet. Porém, ela não foi escolhida como solução principal porque
o problema já apresenta desempenho excelente com modelos menores, mais rápidos e mais fáceis de
explicar em um trabalho acadêmico de disciplina.

Justificativas:

- custo computacional maior;
- maior complexidade para explicar e reproduzir;
- ganho potencial pequeno diante do desempenho já obtido.
