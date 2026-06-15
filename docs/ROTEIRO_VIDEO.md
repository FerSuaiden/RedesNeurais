# Roteiro para Vídeo de 10 Minutos

Uma apresentação de 10 minutos funciona melhor quando cada parte já entra com
tempo previsto. A sugestão abaixo distribui o conteúdo de forma equilibrada e
já está alinhada com a versão mais recente do projeto.

Nos primeiros 1 minuto, vale apresentar o problema. A fala pode explicar que a
mosca-branca é uma praga agrícola relevante e que o objetivo do trabalho foi
classificar recortes de insetos em duas classes: `WF`, correspondente à
mosca-branca, e `MR`, correspondente aos demais insetos presentes nas
armadilhas.

Entre 1:00 e 2:30, a apresentação pode mostrar a base de dados. Nesse trecho,
é importante explicar que o conjunto contém recortes derivados de imagens
originais de armadilhas adesivas amarelas, comentar o desbalanceamento entre as
classes e mostrar alguns exemplos visuais. Esse é um bom momento para destacar
que os recortes de `WF` costumam ser menores, enquanto os de `MR` geralmente
ocupam mais área.

Entre 2:30 e 4:00, convém explicar a revisão metodológica. Essa é uma parte
importante do vídeo, porque diferencia a entrega final de uma solução mais
ingênua. O ponto principal é mostrar que uma divisão por recorte gerava
vazamento entre treino e teste, já que diferentes recortes podiam vir da mesma
imagem original. Em seguida, basta explicar que o protocolo final foi corrigido
com divisão por imagem de origem.

Entre 4:00 e 5:30, a apresentação pode cobrir a modelagem. O ideal é explicar
de forma curta por que três abordagens foram avaliadas: um baseline geométrico,
um MLP e uma CNN. O baseline geométrico serve para medir o quanto o problema já
é explicável por largura, altura, área e razão de aspecto. O MLP funciona como
comparação neural direta. A CNN é a arquitetura principal porque explora melhor
relações locais da imagem, como contornos, regiões compactas e formas
alongadas.

Entre 5:30 e 7:00, vale apresentar o pré-processamento. Nesse ponto, a fala
pode destacar o uso de padding para preservar a proporção do inseto antes do
redimensionamento, além do uso de rotações leves, espelhamento horizontal e
normalização dos canais. Também cabe mencionar o `WeightedRandomSampler`, usado
para reduzir o efeito do desbalanceamento durante o treino.

Entre 7:00 e 8:30, a apresentação deve entrar nos resultados. Esse trecho deve
mostrar que o baseline geométrico já alcança desempenho muito alto, o que
revela um atalho forte no dataset. Em seguida, convém comparar MLP e CNN,
explicando que a `Average Precision` é alta nos dois casos, mas que a CNN
apresenta comportamento operacional melhor para a classe `MR`.

Entre 8:30 e 9:30, a fala pode explicar a interpretação das métricas. Aqui vale
esclarecer a diferença entre `loss`, `Average Precision`, precisão e revocação
no threshold 0,5. Também é importante dizer explicitamente que revocação igual
a `1,0000` para `MR` não significa acerto perfeito em tudo; significa apenas
que, naquele teste e naquele threshold, todas as amostras reais de `MR` foram
recuperadas.

No último minuto, entre 9:30 e 10:00, a conclusão pode sintetizar a mensagem
principal do trabalho. A formulação sugerida é a seguinte: o dataset é muito
separável por geometria, o protocolo precisou ser corrigido para evitar
vazamento, e mesmo em uma avaliação mais rigorosa a CNN permaneceu como a melhor
escolha entre os modelos neurais avaliados.
