"""
Jogo de Adivinhação de Moscas - Servidor Flask local
Apresenta imagens reais do dataset Embrapa (5 fotos por rodada).
Exemplos difíceis = bounding box pequeno (inseto quase invisível) ou truncado.
"""

import os
import random
import xml.etree.ElementTree as ET
import glob
from flask import Flask, jsonify, render_template, send_from_directory

app = Flask(__name__)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MR_DIR    = os.path.join(BASE_DIR, "dataset-embrapa", "MR")
WF_DIR    = os.path.join(BASE_DIR, "dataset-embrapa", "WF")
LABEL_DIR = os.path.join(BASE_DIR, "dataset-embrapa", "labels")

# ── Catalogar imagens (normal vs. difícil) ─────────────────────────────────
# Difícil: bbox pequeno (<= 50 px em qualquer dimensão) OU truncated=1
HARD_THRESHOLD_PX = 50

normal: dict[str, list[str]] = {"MR": [], "WF": []}
hard:   dict[str, list[str]] = {"MR": [], "WF": []}

def _catalogar():
    for xml_path in glob.glob(os.path.join(LABEL_DIR, "*.xml")):
        stem = os.path.splitext(os.path.basename(xml_path))[0]
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        counters: dict[str, int] = {}
        for obj in root.findall("object"):
            name_el = obj.find("name")
            if name_el is None or name_el.text not in ("MR", "WF"):
                continue
            cls = name_el.text
            counters[cls] = counters.get(cls, 0) + 1
            idx = counters[cls]
            fname = f"{stem}_{cls}_{idx:03d}.jpg"
            # Verificar se o arquivo realmente existe
            folder = MR_DIR if cls == "MR" else WF_DIR
            if not os.path.isfile(os.path.join(folder, fname)):
                continue
            # Classificar como difícil ou normal
            trunc_el = obj.find("truncated")
            bb = obj.find("bndbox")
            is_truncated = trunc_el is not None and trunc_el.text == "1"
            is_small = False
            if bb is not None:
                try:
                    w = int(bb.find("xmax").text) - int(bb.find("xmin").text)
                    h = int(bb.find("ymax").text) - int(bb.find("ymin").text)
                    is_small = w <= HARD_THRESHOLD_PX or h <= HARD_THRESHOLD_PX
                except (TypeError, ValueError):
                    pass
            if is_truncated or is_small:
                hard[cls].append(fname)
            else:
                normal[cls].append(fname)

_catalogar()

# Garante que haja imagens suficientes em ambas as listas
for cls in ("MR", "WF"):
    if len(hard[cls]) < 10:           # poucos difíceis → complementa com normais
        hard[cls] += normal[cls][:200]

def _sample5(cls: str, modo: str) -> list[str]:
    """Sorteia 5 imagens distintas da classe e modo pedidos."""
    pool = hard[cls] if modo == "hard" else normal[cls]
    if len(pool) < 5:
        pool = (hard[cls] + normal[cls])
    return random.sample(pool, min(5, len(pool)))


# ── Rotas ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/round")
def get_round():
    """Retorna 5 imagens da mesma classe para uma rodada do jogo."""
    from flask import request
    modo = request.args.get("modo", "normal")   # normal | hard
    fly_class = random.choice(["MR", "WF"])
    filenames = _sample5(fly_class, modo)
    urls = [f"/img/{fly_class}/{fn}" for fn in filenames]
    return jsonify({"urls": urls, "classe": fly_class})


@app.route("/img/<classe>/<filename>")
def serve_image(classe, filename):
    """Serve as imagens do dataset com validação de segurança."""
    # Previne path traversal
    filename = os.path.basename(filename)
    if classe == "MR":
        return send_from_directory(MR_DIR, filename)
    elif classe == "WF":
        return send_from_directory(WF_DIR, filename)
    return "Not found", 404


if __name__ == "__main__":
    print("=" * 55)
    print("🪰  Jogo das Moscas  🪰")
    print("=" * 55)
    print(f"   MR normais  : {len(normal['MR']):>5,}  |  difíceis: {len(hard['MR']):>4,}")
    print(f"   WF normais  : {len(normal['WF']):>5,}  |  difíceis: {len(hard['WF']):>4,}")
    print()
    print("   Abra no navegador: http://localhost:5000")
    print("=" * 55)
    app.run(debug=False, port=5000)
