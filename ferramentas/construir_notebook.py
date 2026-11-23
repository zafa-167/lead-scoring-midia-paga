"""Converte notebook_fonte.py (formato percent) em .ipynb e executa todas as células."""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
fonte = (AQUI / "notebook_fonte.py").read_text(encoding="utf-8")

celulas = []
for bloco in fonte.split("\n# %%"):
    bloco = bloco.lstrip("# %").rstrip() if not celulas and bloco.startswith("# %%") else bloco.rstrip()
    if not bloco.strip():
        continue
    cabecalho, _, corpo = bloco.partition("\n")
    if "[markdown]" in cabecalho:
        linhas = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in corpo.split("\n")]
        celulas.append(nbformat.v4.new_markdown_cell("\n".join(linhas).strip()))
    else:
        celulas.append(nbformat.v4.new_code_cell(corpo.strip()))

nb = nbformat.v4.new_notebook(cells=celulas)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}

destino = RAIZ / "02_Notebook" / "Lead_Scoring_P2.ipynb"
if "--sem-executar" not in sys.argv:
    NotebookClient(nb, timeout=3600, kernel_name="python3",
                   resources={"metadata": {"path": str(destino.parent)}}).execute()
nbformat.write(nb, destino)
print("Notebook gravado em", destino, "com", len(celulas), "células")
