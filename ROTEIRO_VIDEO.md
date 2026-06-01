# Roteiro Sugerido para o Vídeo de 10 Minutos

## 1. Contexto do problema

- explicar rapidamente o impacto da mosca-branca na agricultura;
- dizer que o objetivo é distinguir `WF` de `MR` em recortes de insetos.

## 2. Base de dados

- mostrar a estrutura das pastas `WF` e `MR`;
- comentar o desbalanceamento entre as classes;
- mostrar alguns exemplos visuais.

## 3. Escolha dos tópicos da disciplina

- justificar CNN como tema principal;
- explicar por que usar MLP como baseline;
- comentar por que transferência de aprendizado foi considerada, mas não foi a principal escolha.

## 4. Pré-processamento e divisão

- explicar redimensionamento;
- explicar divisão treino/validação/teste estratificada;
- mencionar o uso de `WeightedRandomSampler`.

## 5. Modelos

- apresentar o MLP baseline;
- apresentar a CNN principal;
- comentar quantidade de parâmetros de cada um.

## 6. Resultados

- mostrar a curva Precisão-Revocação;
- mostrar a métrica Average Precision;
- comparar MLP e CNN.

## 7. Conclusão

- destacar que o problema foi resolvido com AP acima de 0,99;
- reforçar que a CNN foi escolhida por ser mais adequada para visão computacional;
- comentar possíveis trabalhos futuros.
