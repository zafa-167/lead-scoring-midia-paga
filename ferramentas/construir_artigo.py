"""Gera o artigo científico (Word com equações editáveis e PDF) a partir de artefatos/resultados.json.

Os números do texto são interpolados do JSON gravado pelo notebook, para que artigo,
notebook e dashboard fiquem sempre consistentes.
"""
import json
import re
import subprocess
from pathlib import Path

import pypandoc
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
SAIDA = RAIZ / "01_Artigo"
SAIDA.mkdir(exist_ok=True)
R = json.loads((RAIZ / "artefatos" / "resultados.json").read_text(encoding="utf-8"))
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


# ------------------------------------------------------------------ formatação
def n(valor, casas=3):
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def i(valor):
    return f"{int(round(valor)):,}".replace(",", ".")


def p(valor, casas=1):
    return n(valor * 100, casas) + "%"


def tabela(cabecalho, linhas, legenda, alinhamento=None):
    alinhamento = alinhamento or ["l"] + ["r"] * (len(cabecalho) - 1)
    marca = {"l": ":---", "r": "---:", "c": ":---:"}
    saida = ["| " + " | ".join(cabecalho) + " |", "|" + "|".join(marca[a] for a in alinhamento) + "|"]
    saida += ["| " + " | ".join(str(c) for c in linha) + " |" for linha in linhas]
    return "\n".join(saida) + f"\n\n: {legenda}\n"


# ------------------------------------------------------------------ valores
m, ic = R["teste"]["metricas"], R["teste"]["ic95"]
val = R["comparacao_modelos_validacao"]
est = {(e["algoritmo"], e["estrategia"]): e for e in R["comparacao_estrategias_validacao"]}
cal = R["calibracao_validacao"]
decis = R["decis_teste"]
pol = {round(c["capacidade"], 2): c for c in R["politica"]["cenarios"]}
cons = {round(c["capacidade"], 2): c for c in R["politica_conservadora"]}
abl = R["ablacao"]
rt = R["robustez_temporal"]
hp = R["hiperparametros"]
trat = R["tratamento"]
shap_imp = R["shap_importancia"]
shap_total = sum(shap_imp.values())
macro = ["taxa_variacao_emprego", "indice_precos_consumidor", "indice_confianca_consumidor"]
rc = {**R["razao_chances"]["maiores"], **R["razao_chances"]["menores"]}
lim = R["teste"]["limiar"]
conv_teste = R["particoes"]["teste"]["conversoes"]

V = {
    "n_bruto": i(R["base"]["registros_brutos"]), "n_tratado": i(R["base"]["registros_tratados"]),
    "taxa_conv": p(R["base"]["taxa_conversao"], 2), "razao": n(R["base"]["razao_desbalanceamento"], 2),
    "n_pred": R["base"]["n_preditoras"], "n_matriz": R["base"]["n_colunas_matriz"],
    "n_unknown": i(trat["registros_com_unknown"]),
    "pct_unknown": p(trat["registros_com_unknown"] / R["base"]["registros_tratados"]),
    "n_sentinela": i(trat["sentinela_999"]),
    "pct_sentinela": p(trat["sentinela_999"] / R["base"]["registros_tratados"]),
    "n_wins": i(trat["registros_winsorizados"]),
    "vif_antes": n(max(v for v in R["vif"]["VIF antes"].values() if v == v), 2),
    "vif_depois": n(max(v for v in R["vif"]["VIF depois"].values() if v == v), 2),
    "n_treino": i(R["particoes"]["treino"]["registros"]), "n_val": i(R["particoes"]["validação"]["registros"]),
    "n_teste": i(R["particoes"]["teste"]["registros"]), "conv_teste": i(conv_teste),
    "prev": n(R["teste"]["prevalencia"]),
    "auc_pr": n(m["AUC-PR"]), "auc_pr_lo": n(ic["AUC-PR"][0]), "auc_pr_hi": n(ic["AUC-PR"][1]),
    "auc_roc": n(m["AUC-ROC"]), "auc_roc_lo": n(ic["AUC-ROC"][0]), "auc_roc_hi": n(ic["AUC-ROC"][1]),
    "rec20": p(m["Revocação top 20%"]), "rec20_lo": p(ic["Revocação top 20%"][0]),
    "rec20_hi": p(ic["Revocação top 20%"][1]),
    "lift1": n(m["Lift 1º decil"], 2), "lift1_lo": n(ic["Lift 1º decil"][0], 2),
    "lift1_hi": n(ic["Lift 1º decil"][1], 2),
    "razao_aucpr": n(m["AUC-PR"] / R["teste"]["prevalencia"], 1),
    "brier": n(m["Brier"], 4), "ece": n(m["ECE"], 4), "ece_antes": n(m["ECE sem calibração"], 4),
    "brier_antes": n(m["Brier sem calibração"], 4),
    "dec1_taxa": p(decis[0]["taxa_conversao"]), "dec1_prev": p(decis[0]["prob_media_prevista"]),
    "dec2_taxa": p(decis[1]["taxa_conversao"]), "dec10_taxa": p(decis[9]["taxa_conversao"]),
    "ganho40": p(decis[3]["ganho_acumulado"]), "ganho50": p(decis[4]["ganho_acumulado"]),
    "ganho30": p(decis[2]["ganho_acumulado"]),
    "red_cac_10": p(pol[0.1]["reducao_cac_pct"] / 100), "red_cac_20": p(pol[0.2]["reducao_cac_pct"] / 100),
    "red_cac_40": p(pol[0.4]["reducao_cac_pct"] / 100),
    "conv20_modelo": i(pol[0.2]["conversoes_modelo"]), "conv20_chegada": i(pol[0.2]["conversoes_ordem_chegada"]),
    "cac20_modelo": n(pol[0.2]["cac_modelo"], 2), "cac20_chegada": n(pol[0.2]["cac_ordem_chegada"], 2),
    "cons20_rec": p(cons[0.2]["conversoes_capturadas_intracontexto"]),
    "cons20_lift": n(cons[0.2]["lift_intracontexto"], 2),
    "cons20_red": p(cons[0.2]["reducao_cac_intracontexto_pct"] / 100),
    "cons40_rec": p(cons[0.4]["conversoes_capturadas_intracontexto"]),
    "cons40_red": p(cons[0.4]["reducao_cac_intracontexto_pct"] / 100),
    "n_grupos": R["n_grupos_contexto"],
    "auc_intra": n(abl[0]["AUC-ROC intracontexto"]),
    "auc_sem_macro": n(abl[1]["AUC-ROC"]), "auc_perfil": n(abl[2]["AUC-ROC"]),
    "aucpr_perfil": n(abl[2]["AUC-PR"]), "rec20_perfil": p(abl[2]["Revocação top 20%"]),
    "rt_auc": n(rt["AUC-ROC"]), "rt_aucpr": n(rt["AUC-PR"]), "rt_prev": n(rt["taxa_conversao_teste"]),
    "rt_prev_treino": n(rt["taxa_conversao_treino"]), "rt_rec20": p(rt["Revocação top 20%"]),
    "rt_lift": n(rt["Lift 1º decil"], 2), "rt_prob": p(rt["prob_media_prevista"]),
    "rt_taxa": p(rt["taxa_conversao_teste"]), "rt_taxa_treino": p(rt["taxa_conversao_treino"]),
    "rt_brier": n(rt["Brier"], 3),
    "shap_macro_pct": p(sum(shap_imp[v] for v in macro) / shap_total, 0),
    "shap_emprego_pct": p(shap_imp["taxa_variacao_emprego"] / shap_total, 0),
    "shap_emprego": n(shap_imp["taxa_variacao_emprego"]),
    "rc_previo": n(rc["contato_previo"], 2), "rc_sucesso": n(rc["resultado_campanha_anterior_success"], 2),
    "rc_emprego": n(rc["taxa_variacao_emprego"], 2), "rc_telefone": n(rc["canal_contato_telephone"], 2),
    "rc_marco": n(rc["mes_contato_mar"], 2), "rc_maio": n(rc["mes_contato_may"], 2),
    "C_rl": n(hp["regressao_logistica"]["C"], 1),
    "limiar": n(lim["limiar"]), "lim_prec": p(lim["precisao"]), "lim_rev": p(lim["revocacao"]),
    "lim_f1": n(lim["f1"]), "lim_abord": p(lim["pct_leads_abordados"]),
    "xgb_val_aucpr": n(est[("XGBoost", "Sem tratamento")]["AUC-PR"]),
    "rl_val_aucpr": n(val["Regressão logística"]["AUC-PR"]), "rf_val_aucpr": n(val["Floresta aleatória"]["AUC-PR"]),
    "prob_pond": p(est[("XGBoost", "Ponderação de classes")]["Prob. média prevista"]),
    "prob_smote": p(est[("XGBoost", "SMOTE")]["Prob. média prevista"]),
    "prob_sem": p(est[("XGBoost", "Sem tratamento")]["Prob. média prevista"]),
    "brier_pond": n(est[("XGBoost", "Ponderação de classes")]["Brier"]),
    "brier_sem": n(est[("XGBoost", "Sem tratamento")]["Brier"]),
    "f1_sem_05": n(est[("XGBoost", "Sem tratamento")]["F1 (limiar 0,5)"]),
    "f1_sem_ot": n(est[("XGBoost", "Sem tratamento")]["F1 (limiar ótimo)"]),
    "f1_pond_ot": n(est[("XGBoost", "Ponderação de classes")]["F1 (limiar ótimo)"]),
    "f1_smote_ot": n(est[("XGBoost", "SMOTE")]["F1 (limiar ótimo)"]),
    "aucpr_smote": n(est[("XGBoost", "SMOTE")]["AUC-PR"]),
    "cpl": n(R["politica"]["cpl"], 2), "custo_abord": n(R["politica"]["custo_abordagem"], 2),
}

# ------------------------------------------------------------------ tabelas
assoc = {a["variavel"]: a for a in R["associacao"]}
nomes_assoc = [("resultado_campanha_anterior", "Preditiva"), ("mes_contato", "Preditiva"),
               ("profissao", "Preditiva"), ("canal_contato", "Preditiva"), ("inadimplencia", "Preditiva"),
               ("escolaridade", "Preditiva"), ("estado_civil", "Preditiva"),
               ("dia_semana", "Excluída (efeito desprezível)"),
               ("financ_imobiliario", "Excluída (efeito desprezível)"),
               ("emprestimo_pessoal", "Excluída (efeito desprezível)"),
               ("duracao_contato_s", "Excluída (vazamento)"), ("n_empregados", "Excluída (VIF)"),
               ("euribor_3m", "Excluída (VIF)"), ("taxa_variacao_emprego", "Preditiva"),
               ("indice_precos_consumidor", "Preditiva"), ("n_contatos_anteriores", "Preditiva"),
               ("n_contatos_campanha", "Preditiva"), ("indice_confianca_consumidor", "Preditiva"),
               ("idade", "Preditiva")]
V["tab_assoc"] = tabela(
    ["Variável", "Medida", "Tamanho de efeito", "Papel no modelo"],
    [[f"`{v}`", assoc[v]["medida"], n(assoc[v]["efeito"]), papel] for v, papel in nomes_assoc],
    "Tabela 3 — Associação de cada variável com a conversão e papel na modelagem", ["l", "l", "r", "l"])

V["tab_modelos"] = tabela(
    ["Modelo", "AUC-PR (CV no treino)", "AUC-PR", "AUC-ROC", "Revocação top 20%", "Lift 1º decil", "Brier", "ECE"],
    [[nome, n(v["AUC-PR CV (treino)"]), n(v["AUC-PR"]), n(v["AUC-ROC"]), p(v["Revocação top 20%"]),
      n(v["Lift 1º decil"], 2), n(v["Brier"], 4), n(v["ECE"], 4)] for nome, v in val.items()],
    "Tabela 5 — Comparação dos modelos na partição de validação")

V["tab_estrategias"] = tabela(
    ["Algoritmo", "Estratégia", "AUC-PR", "AUC-ROC", "Brier", "Prob. média prevista", "F1 (limiar 0,5)",
     "Limiar ótimo", "F1 (limiar ótimo)"],
    [[a, e, n(v["AUC-PR"]), n(v["AUC-ROC"]), n(v["Brier"]), p(v["Prob. média prevista"]),
      n(v["F1 (limiar 0,5)"]), n(v["Limiar ótimo"]), n(v["F1 (limiar ótimo)"])] for (a, e), v in est.items()],
    "Tabela 6 — Estratégias para o desbalanceamento, avaliadas na validação", ["l", "l"] + ["r"] * 7)

V["tab_calibracao"] = tabela(
    ["Método", "Brier", "ECE", "Perda logarítmica"],
    [[nome, n(v["Brier"], 5), n(v["ECE"], 5), n(v["Log-loss"], 5)] for nome, v in cal.items()],
    "Tabela 7 — Calibração do modelo final (validação cruzada dentro da validação)")

V["tab_teste"] = tabela(
    ["Métrica", "Valor no teste", "IC 95% (bootstrap)", "Referência sem modelo"],
    [["AUC-PR", V["auc_pr"], f"{V['auc_pr_lo']} a {V['auc_pr_hi']}", V["prev"]],
     ["AUC-ROC", V["auc_roc"], f"{V['auc_roc_lo']} a {V['auc_roc_hi']}", "0,500"],
     ["Revocação nos 20% do topo", V["rec20"], f"{V['rec20_lo']} a {V['rec20_hi']}", "20,0%"],
     ["Lift no 1º decil", V["lift1"], f"{V['lift1_lo']} a {V['lift1_hi']}", "1,00"],
     ["Brier (calibrado)", V["brier"], "", n(R["teste"]["prevalencia"] * (1 - R["teste"]["prevalencia"]), 4)],
     ["ECE (calibrado)", V["ece"], "", ""]],
    "Tabela 8 — Desempenho do modelo final no conjunto de teste")

cand = R["teste"]["candidatos"]
V["tab_candidatos"] = tabela(
    ["Modelo", "AUC-PR", "AUC-ROC", "Revocação top 20%", "Lift 1º decil", "Brier"],
    [[nome, n(v["AUC-PR"]), n(v["AUC-ROC"]), p(v["Revocação top 20%"]), n(v["Lift 1º decil"], 2), n(v["Brier"], 4)]
     for nome, v in cand.items()],
    "Tabela 9 — Os três algoritmos no conjunto de teste (registro; a escolha foi feita na validação)")

V["tab_decis"] = tabela(
    ["Decil", "Leads", "Conversões", "Taxa de conversão", "Probabilidade média prevista", "Lift", "Ganho acumulado"],
    [[d["decil"], i(d["leads"]), i(d["conversoes"]), p(d["taxa_conversao"]), p(d["prob_media_prevista"]),
      n(d["lift"], 2), p(d["ganho_acumulado"])] for d in decis],
    "Tabela 10 — Conversão por decil do score no conjunto de teste (decil 1 = topo da fila)", ["c"] + ["r"] * 6)

V["tab_politica"] = tabela(
    ["Capacidade", "Leads atendidos", "Conversões (ordem de chegada)", "Conversões (fila priorizada)",
     "CAC ordem de chegada (reais)", "CAC fila priorizada (reais)", "Redução do CAC"],
    [[p(c["capacidade"], 0), i(c["leads_atendidos"]), i(c["conversoes_ordem_chegada"]), i(c["conversoes_modelo"]),
      n(c["cac_ordem_chegada"], 2), n(c["cac_modelo"], 2), p(c["reducao_cac_pct"] / 100)]
     for c in R["politica"]["cenarios"]],
    "Tabela 12 — Política de priorização em fila única, por cenário de capacidade (conjunto de teste)",
    ["c"] + ["r"] * 6)

V["tab_conservador"] = tabela(
    ["Capacidade", "Conversões capturadas (fila única)", "Conversões capturadas (intracontexto)",
     "Lift (fila única)", "Lift (intracontexto)", "Redução do CAC (fila única)", "Redução do CAC (intracontexto)"],
    [[p(c["capacidade"], 0), p(c["conversoes_capturadas_fila_unica"]), p(c["conversoes_capturadas_intracontexto"]),
      n(c["lift_fila_unica"], 2), n(c["lift_intracontexto"], 2), p(c["reducao_cac_fila_unica_pct"] / 100),
      p(c["reducao_cac_intracontexto_pct"] / 100)] for c in R["politica_conservadora"]],
    "Tabela 13 — Teto (fila única) e piso (intracontexto) do ganho de priorização", ["c"] + ["r"] * 6)

V["tab_ablacao"] = tabela(
    ["Conjunto de variáveis", "Nº de variáveis", "AUC-PR", "AUC-ROC", "Revocação top 20%", "AUC-ROC intracontexto"],
    [[a["Conjunto de variáveis"], a["Nº de variáveis"], n(a["AUC-PR"]), n(a["AUC-ROC"]),
      p(a["Revocação top 20%"]), n(a["AUC-ROC intracontexto"])] for a in abl],
    "Tabela 11 — Ablação de variáveis no conjunto de teste")

rf, xg = hp["floresta_aleatoria"], hp["xgboost"]
V["tab_hiper"] = tabela(
    ["Modelo", "Espaço de busca", "Configuração selecionada"],
    [["Regressão logística", "C em {0,001; 0,003; 0,01; 0,03; 0,1; 0,3; 1; 3; 10}; ponderação de classes",
      f"C = {V['C_rl']}"],
     ["Floresta aleatória (12 sorteios)",
      "árvores em {300, 500}; profundidade em {6, 8, 10, 14, livre}; mínimo por folha em {5, 10, 20, 40}; "
      "atributos por divisão em {raiz, 0,3, 0,5}",
      f"{rf['n_estimators']} árvores; profundidade {rf['max_depth']}; mínimo de {rf['min_samples_leaf']} por folha; "
      f"atributos por divisão: {'raiz quadrada' if rf['max_features'] == 'sqrt' else rf['max_features']}"],
     ["XGBoost (30 sorteios)",
      "árvores em {200, 300, 400, 600}; taxa de aprendizado em {0,01 a 0,1}; profundidade em {2 a 6}; "
      "peso mínimo por folha em {1, 5, 10, 20}; subamostra em {0,6; 0,8; 1}; colunas por árvore em {0,5; 0,7; 1}; "
      "λ em {1, 5, 10}; γ em {0, 1, 5}",
      f"{xg['n_estimators']} árvores; η = {n(xg['learning_rate'], 2)}; profundidade {xg['max_depth']}; "
      f"peso mínimo {xg['min_child_weight']}; subamostra {n(xg['subsample'], 1)}; "
      f"colunas {n(xg['colsample_bytree'], 1)}; λ = {xg['reg_lambda']}; γ = {xg['gamma']}"]],
    "Tabela 4 — Espaços de busca e hiperparâmetros selecionados por validação cruzada", ["l", "l", "l"])

# ------------------------------------------------------------------ texto
texto = (AQUI / "artigo_modelo.md").read_text(encoding="utf-8")
faltando = sorted(set(re.findall(r"@([a-zA-Z0-9_]+)@", texto)) - set(V))
assert not faltando, f"Marcadores sem valor: {faltando}"
texto = re.sub(r"@([a-zA-Z0-9_]+)@", lambda mo: str(V[mo.group(1)]), texto)
arquivo_md = SAIDA / "_artigo.md"
arquivo_md.write_text(texto, encoding="utf-8")

# ------------------------------------------------------------------ Word
referencia = AQUI / "_referencia.docx"
referencia.write_bytes(subprocess.run([pypandoc.get_pandoc_path(), "--print-default-data-file", "reference.docx"],
                                      capture_output=True, check=True).stdout)
doc = Document(str(referencia))
for estilo in doc.styles:
    if estilo.type != 1:
        continue
    fonte = estilo.font
    fonte.name = "Times New Roman"
    if estilo.element.rPr is not None and estilo.element.rPr.rFonts is not None:
        for atributo in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
            estilo.element.rPr.rFonts.attrib.pop(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}" + atributo, None)
    if estilo.name.startswith("Heading") or estilo.name in ("Title", "Subtitle", "Author", "Date", "Abstract Title"):
        fonte.color.rgb = RGBColor(0, 0, 0)
        fonte.bold = True
    if estilo.name in ("Normal", "Body Text", "First Paragraph", "Compact", "Abstract", "Bibliography"):
        fonte.size = Pt(12)
    if estilo.name in ("Body Text", "First Paragraph", "Abstract", "Bibliography"):
        estilo.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        estilo.paragraph_format.space_after = Pt(8)
        estilo.paragraph_format.line_spacing = 1.25
    if estilo.name == "Compact":
        fonte.size = Pt(9.5)
    if estilo.name in ("Image Caption", "Table Caption", "Caption"):
        fonte.size, fonte.italic = Pt(10), False
        estilo.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if estilo.name == "Title":
        fonte.size = Pt(16)
        estilo.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if estilo.name == "Heading 1":
        fonte.size = Pt(13)
    if estilo.name == "Heading 2":
        fonte.size = Pt(12)
    if estilo.name == "Heading 3":
        fonte.size, fonte.italic = Pt(12), True
for secao in doc.sections:
    secao.left_margin = secao.right_margin = Pt(72)
    secao.top_margin = secao.bottom_margin = Pt(72)
doc.save(str(referencia))

arquivo_docx = SAIDA / "Artigo_Lead_Scoring_P2.docx"
pypandoc.convert_file(str(arquivo_md), "docx", outputfile=str(arquivo_docx),
                      extra_args=[f"--reference-doc={referencia}", f"--resource-path={RAIZ}",
                                  "--from=markdown+pipe_tables+table_captions+implicit_figures+tex_math_dollars"])

# Bordas nas tabelas e centralização das figuras e equações
doc = Document(str(arquivo_docx))
for t in doc.tables:
    t.style = "Table Grid" if "Table Grid" in [s.name for s in doc.styles] else t.style
    t.alignment = 1
doc.save(str(arquivo_docx))
print("Word gravado:", arquivo_docx)

# ------------------------------------------------------------------ PDF (HTML + Chrome)
CSS = """
@page { size: A4; margin: 2.5cm 2.5cm 2.3cm 2.5cm; }
body { font-family: 'Times New Roman', Times, serif; font-size: 12pt; line-height: 1.4; color: #000;
       text-align: justify; max-width: none; margin: 0; padding: 0; hyphens: auto; }
h1.title { font-size: 17pt; text-align: center; margin: 0 0 10pt 0; line-height: 1.25; }
h1 { font-size: 13pt; margin: 18pt 0 6pt 0; text-align: left; }
h2 { font-size: 12pt; margin: 14pt 0 4pt 0; text-align: left; }
h3 { font-size: 12pt; font-style: italic; margin: 10pt 0 3pt 0; text-align: left; }
p { margin: 0 0 8pt 0; }
table { border-collapse: collapse; margin: 6pt auto 12pt auto; font-size: 9pt; line-height: 1.25; width: 100%; }
th, td { border: 0.6pt solid #666; padding: 2.5pt 4pt; text-align: left; vertical-align: top; hyphens: none; }
th { background: #E9EEF4; }
caption { font-size: 10pt; margin-bottom: 4pt; caption-side: top; font-weight: bold; text-align: center; }
figure { margin: 10pt 0 12pt 0; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-size: 10pt; font-weight: bold; margin-top: 3pt; text-align: center; }
table, tr { break-inside: avoid; }
h1, h2, h3 { break-after: avoid; }
code { font-family: Consolas, monospace; font-size: 9.5pt; }
td code, th code { font-size: 8.5pt; }
p.author { text-align: center; margin: 0 0 2pt 0; font-size: 11pt; }
p.author:first-of-type { font-size: 12pt; font-weight: bold; }
math[display="block"] { margin: 8pt 0; }
header#title-block-header { margin-bottom: 4pt; }
"""
(SAIDA / "_estilo.css").write_text(CSS, encoding="utf-8")
arquivo_html = SAIDA / "_artigo.html"
pypandoc.convert_file(str(arquivo_md), "html5", outputfile=str(arquivo_html),
                      extra_args=["--standalone", "--mathml", "--embed-resources", f"--resource-path={RAIZ}",
                                  f"--css={SAIDA / '_estilo.css'}", "--metadata=lang:pt-BR",
                                  "--from=markdown+pipe_tables+table_captions+implicit_figures+tex_math_dollars"])
arquivo_pdf = SAIDA / "Artigo_Lead_Scoring_P2.pdf"
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={arquivo_pdf}", arquivo_html.as_uri()], check=True, capture_output=True)
for temporario in (arquivo_md, arquivo_html, SAIDA / "_estilo.css", referencia):
    temporario.unlink(missing_ok=True)
print("PDF gravado:", arquivo_pdf)
