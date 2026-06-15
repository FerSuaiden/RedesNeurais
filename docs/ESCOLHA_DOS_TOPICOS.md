# Escolha dos Tópicos da Disciplina

## Tema principal: Redes Neurais Convolucionais

Redes neurais convolucionais foram escolhidas como eixo principal porque a entrada do problema é
visual. Em uma imagem, pixels próximos não são independentes: eles compõem bordas, regiões claras,
contornos, partes compactas e formas alongadas. Esse arranjo local é o que se chama aqui de
estrutura espacial da imagem. Uma CNN explora essa organização por meio de filtros locais, o que a
torna mais adequada do que uma rede densa para modelar padrões desse tipo.

No conjunto utilizado, essa escolha permaneceu tecnicamente justificável mesmo após a revisão
metodológica com split por imagem original. A CNN apresentou desempenho muito alto e, além disso,
teve comportamento operacional melhor do que o MLP para a classe `MR`.

## Uso do MLP como baseline

O MLP foi mantido como baseline porque ele oferece uma comparação direta com uma arquitetura que
recebe os mesmos dados, mas sem explorar explicitamente relações locais entre pixels vizinhos. Isso
permite observar o quanto do desempenho vem de uma modelagem visual mais apropriada e o quanto já
pode ser obtido com uma rede densa sobre pixels achatados.

## Observação metodológica importante

Durante a revisão do experimento, ficou claro que o dataset possui um atalho geométrico forte.
Largura, altura, área e razão de aspecto dos recortes já explicam grande parte do desempenho. Por
isso, a avaliação final foi organizada de modo mais rigoroso: o split passou a ser feito por imagem
original, foi incluído um baseline geométrico como controle e a análise deixou de depender apenas
de Average Precision.
