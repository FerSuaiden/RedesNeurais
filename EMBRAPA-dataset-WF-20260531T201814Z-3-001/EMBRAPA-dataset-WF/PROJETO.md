# Dataset Moscas — Visão Geral do Projeto

## Estrutura de Pastas

```
dataset-moscas/
├── imagens/                284 fotos originais (.jpg), alta resolução (5184x3456 px)
├── labels/                 284 arquivos de anotação (.xml), um por imagem, formato Pascal VOC
├── WF/                     6350 recortes individuais de objetos WF
├── MR/                     1287 recortes individuais de objetos MR
├── visualize_boxes.ipynb   Notebook Jupyter para visualizar e recortar bounding boxes
└── PROJETO.md              este arquivo
```

> **Observação:** `labels/` e `imagens/` têm 284 arquivos cada — paridade total.

---

## Classes

| Classe | Descrição |
|--------|-----------|
| **WF** | Mosca-branca (*Bemisia tabaci*) — inseto pequeno, bounding box pequeno |
| **MR** | Mosca do mediterrâneo ou similar — inseto maior, bounding box maior |

---

## Formato do Arquivo XML (Pascal VOC)

Cada arquivo `.xml` corresponde a uma imagem e pode conter múltiplos objetos anotados.

```xml
<annotation>
  <filename>1000.jpg</filename>

  <size>
    <width>5184</width>    <!-- largura da imagem em pixels -->
    <height>3456</height>  <!-- altura da imagem em pixels -->
    <depth>3</depth>       <!-- canais de cor (RGB) -->
  </size>

  <object>
    <name>WF</name>           <!-- classe do objeto: WF ou MR -->
    <truncated>0</truncated>  <!-- 1 = objeto cortado na borda da imagem -->
    <difficult>0</difficult>  <!-- 1 = difícil de identificar (pode ser ignorado no treino) -->
    <bndbox>
      <xmin>60</xmin>         <!-- borda esquerda do box (pixels a partir da esquerda) -->
      <ymin>692</ymin>        <!-- borda superior do box (pixels a partir do topo) -->
      <xmax>98</xmax>         <!-- borda direita do box -->
      <ymax>737</ymax>        <!-- borda inferior do box -->
    </bndbox>
  </object>

  <!-- mais blocos <object> ... -->
</annotation>
```

---

## Coordenadas do Bounding Box

A origem `(0, 0)` fica no canto superior esquerdo da imagem. O eixo X cresce para a direita, o eixo Y cresce para baixo.

```
  (xmin, ymin) ----------- (xmax, ymin)
       |                         |
       |         objeto          |
       |                         |
  (xmin, ymax) ----------- (xmax, ymax)
```

| Operação | Fórmula |
|----------|---------|
| Recorte (PIL) | `img.crop((xmin, ymin, xmax, ymax))` |
| Largura do box | `xmax - xmin` |
| Altura do box | `ymax - ymin` |
| Centro do box | `((xmin+xmax)/2, (ymin+ymax)/2)` |

- Boxes **WF** são pequenos (~30–80 px de largura)
- Boxes **MR** são maiores (~100–200 px de largura)

---

