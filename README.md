# SCC0270 - Classificação de Moscas

Projeto da disciplina **Redes Neurais e Aprendizado Profundo** (USP/ICMC, 2026-1).

## Problema

Classificação binária de recortes de insetos capturados em armadilhas adesivas amarelas:

| Classe | Descrição |
|--------|-----------|
| `WF`   | Mosca-branca (*Bemisia tabaci*) |
| `MR`   | Demais insetos (*Macrolophus*, *Nesidiocoris*, outros) |

## Estrutura do repositório

```
RedesNeurais/
├── Classificacao_de_Moscas_Entrega.ipynb          # ← Notebook unificado de entrega
├── Classificacao_de_Moscas_Entrega.executed.ipynb # Versão com outputs salvos
├── README.md
├── docs/
│   ├── RELATORIO_CLASSIFICACAO_DE_MOSCAS.tex      # Relatório em LaTeX (SBC)
│   ├── RELATORIO_CLASSIFICACAO_DE_MOSCAS.md       # Relatório em Markdown
│   ├── ROTEIRO_VIDEO.md                           # Roteiro do vídeo de 10 min
│   ├── ESCOLHA_DOS_TOPICOS.md                     # Justificativa dos tópicos
│   ├── sample.tex                                 # Template SBC
│   ├── gerar_notebook_moscas.py                   # Script auxiliar de geração
│   ├── url_video.txt                              # URL do vídeo YouTube
│   └── 05 - Classificação de Moscas.pdf           # Enunciado original
├── images/
│   ├── amostras_moscas.png                        # Exemplos de recortes WF/MR
│   └── distribuicao_recortes.png                  # Distribuição de classes/dimensões
└── EMBRAPA-dataset-WF-*/
    └── EMBRAPA-dataset-WF/
        ├── imagens/   # Fotos originais das armadilhas
        ├── labels/    # Bounding boxes em XML (Pascal VOC)
        ├── WF/        # Recortes de mosca-branca
        └── MR/        # Recortes dos demais insetos
```

## Notebook unificado

O arquivo `Classificacao_de_Moscas_Entrega.ipynb` cobre **todas as tarefas do enunciado**:

| # | Tarefa |
|---|--------|
| 0 | Exploração do dataset original (bounding boxes + geração dos recortes) |
| 1 | Montagem dos dados (leitura de `WF/` e `MR/`) |
| 2 | Análise exploratória (distribuição de classes e geometria) |
| 3 | Divisão treino/validação/teste **sem vazamento** (split por imagem de origem) |
| 4 | Pré-processamento (padding, resize, augmentation, normalização) |
| 5 | DataLoaders com amostragem balanceada (`WeightedRandomSampler`) |
| 6 | Baseline geométrico (Regressão Logística com atributos de bounding box) |
| 7 | Modelos neurais: MLP (baseline) e CNN (modelo principal) |
| 8 | Experimento 1 - MLP |
| 9 | Experimento 2 - CNN |
| 10 | Comparação dos resultados via **Average Precision (PR-AUC)** |
| 11–13 | Discussão, resultados de referência e conclusão |

## Resultados

| Modelo | Test AP |
|--------|---------|
| Baseline geométrico | 0.9994 |
| MLP | 0.9963 |
| CNN | 1.0000 |

## Como reproduzir

### Localmente
```bash
pip install torch torchvision pillow scikit-learn matplotlib seaborn pandas numpy
jupyter notebook Classificacao_de_Moscas_Entrega.ipynb
```

### Google Colab
1. Faça upload do dataset para o Google Drive.
2. Ajuste `DATA_DIR` na célula 4 do notebook para apontar para a pasta montada.
3. Execute todas as células.

## Entregáveis

1. **Notebook** - `Classificacao_de_Moscas_Entrega.ipynb`
2. **Relatório** - `docs/RELATORIO_CLASSIFICACAO_DE_MOSCAS.tex` (formato SBC)
3. **Vídeo** - URL a ser adicionada no arquivo `docs/url_video.txt`

## Dataset

Fonte: [Embrapa](https://www.embrapa.br/) - dataset de insetos em armadilhas adesivas.  
Link original do projeto: <https://drive.google.com/drive/folders/1w1H7Cl9RdQH8ex-UYQ9EFLw0ZIATR18E>

## Referências

1. BAR et al. Downregulation of dystrophin expression in pupae of the whitefly *Bemisia tabaci*. *Insect Mol Biol*, 2019.
2. JIGE; RATNAPARKHE. Population estimation of whitefly for cotton plant. RTEICT, 2017.
3. BARBEDO. Using digital image processing for counting whiteflies. *J. Asia-Pacific Entomology*, 2014.
4. DAVIS; GOADRICH. The relationship between Precision-Recall and ROC curves. ICML, 2006.
