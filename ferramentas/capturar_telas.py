"""Captura telas do dashboard (precisa do Streamlit rodando em localhost:8501)."""
from pathlib import Path
from playwright.sync_api import sync_playwright

DESTINO = Path(__file__).resolve().parent.parent / "figuras" / "telas"
DESTINO.mkdir(parents=True, exist_ok=True)
PAGINAS = {"1. Visão executiva": "tela_visao", "2. Explorador 3D": "tela_3d", "3. Desempenho do modelo": "tela_desempenho",
           "4. Simulador de priorização": "tela3", "5. Pontuar um lead": "tela_lead",
           "6. Fila priorizada": "tela_fila", "7. Assistente de IA": "tela_ia"}

with sync_playwright() as p:
    navegador = p.chromium.launch(channel="chrome", headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    pagina = navegador.new_page(viewport={"width": 1500, "height": 1150}, device_scale_factor=1.5)
    pagina.goto("http://localhost:8501", wait_until="networkidle")
    for nome, arquivo in PAGINAS.items():
        pagina.get_by_text(nome, exact=True).click()
        pagina.wait_for_timeout(7000)
        if nome.startswith("7"):
            pagina.get_by_text("Quanto o CAC cai com 30% de capacidade?", exact=True).click()
            pagina.wait_for_timeout(4000)
        pagina.screenshot(path=str(DESTINO / f"{arquivo}.png"))
        print("ok", nome)
    navegador.close()
