"""Gera a apresentação oral (PPTX) a partir de artefatos/resultados.json e das figuras do notebook."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

RAIZ = Path(__file__).resolve().parent.parent
FIG = RAIZ / "figuras"
R = json.loads((RAIZ / "artefatos" / "resultados.json").read_text(encoding="utf-8"))

AZUL, LARANJA, VERDE, CINZA = RGBColor(0x1F, 0x4E, 0x79), RGBColor(0xE0, 0x7B, 0x39), RGBColor(0x3C, 0x8D, 0x5A), RGBColor(0x5F, 0x6B, 0x7A)
CLARO, BRANCO, PRETO, VERMELHO = RGBColor(0xEA, 0xF0, 0xF7), RGBColor(255, 255, 255), RGBColor(0x22, 0x2B, 0x36), RGBColor(0xC0, 0x39, 0x2B)
FONTE = "Calibri"


def n(v, c=3):
    return f"{v:,.{c}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def i(v):
    return f"{int(round(v)):,}".replace(",", ".")


def p(v, c=1):
    return n(v * 100, c) + "%"


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
EM_BRANCO = prs.slide_layouts[6]
TOTAL = 16
contador = [0]


def caixa(slide, x, y, w, h, texto="", tam=16, cor=PRETO, negrito=False, alinh=PP_ALIGN.LEFT, ancora=MSO_ANCHOR.TOP):
    forma = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = forma.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = ancora
    tf.margin_left = tf.margin_right = Inches(0.05)
    par = tf.paragraphs[0]
    par.alignment = alinh
    r = par.add_run()
    r.text = texto
    r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(tam), negrito, cor, FONTE
    return forma


def retangulo(slide, x, y, w, h, preenchimento, borda=None, forma=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(forma, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = preenchimento
    if borda is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = borda
        s.line.width = Pt(1.25)
    s.shadow.inherit = False
    return s


def novo_slide(rotulo, titulo):
    contador[0] += 1
    slide = prs.slides.add_slide(EM_BRANCO)
    caixa(slide, 0.6, 0.35, 12, 0.3, rotulo.upper(), 11, LARANJA, True)
    caixa(slide, 0.6, 0.62, 12.1, 0.8, titulo, 28, AZUL, True)
    retangulo(slide, 0.65, 1.38, 1.1, 0.045, LARANJA)
    caixa(slide, 0.6, 7.05, 9, 0.3, "Lead Scoring em Mídia Paga · P2 · Projeto Integrador IV · IESB", 9, CINZA)
    caixa(slide, 11.7, 7.05, 1.1, 0.3, f"{contador[0]}/{TOTAL}", 9, AZUL, True, PP_ALIGN.RIGHT)
    return slide


def topicos(slide, itens, x, y, w, h, tam=16, cor=PRETO):
    forma = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = forma.text_frame
    tf.word_wrap = True
    for k, item in enumerate(itens):
        par = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        par.space_after = Pt(7)
        marcador = par.add_run()
        marcador.text = "▪ "
        marcador.font.size, marcador.font.color.rgb, marcador.font.name = Pt(tam), LARANJA, FONTE
        r = par.add_run()
        r.text = item
        r.font.size, r.font.color.rgb, r.font.name = Pt(tam), cor, FONTE
    return forma


def cartao(slide, x, y, w, h, valor, legenda, cor=AZUL):
    retangulo(slide, x, y, w, h, CLARO, cor, MSO_SHAPE.ROUNDED_RECTANGLE)
    caixa(slide, x, y + 0.1, w, h * 0.5, valor, 26, cor, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    caixa(slide, x + 0.1, y + h * 0.55, w - 0.2, h * 0.42, legenda, 11, CINZA, False, PP_ALIGN.CENTER)


def destaque(slide, x, y, w, h, titulo, texto, cor=AZUL, tam=13):
    retangulo(slide, x, y, w, h, CLARO, cor, MSO_SHAPE.ROUNDED_RECTANGLE)
    caixa(slide, x + 0.15, y + 0.08, w - 0.3, 0.3, titulo.upper(), 10, cor, True)
    caixa(slide, x + 0.15, y + 0.38, w - 0.3, h - 0.45, texto, tam, PRETO)


def imagem(slide, arquivo, x, y, w=None, h=None):
    return slide.shapes.add_picture(str(arquivo), Inches(x), Inches(y),
                                    Inches(w) if w else None, Inches(h) if h else None)


def tabela(slide, dados, x, y, w, h, larguras=None, tam=11):
    linhas, colunas = len(dados), len(dados[0])
    t = slide.shapes.add_table(linhas, colunas, Inches(x), Inches(y), Inches(w), Inches(h)).table
    if larguras:
        for c, larg in enumerate(larguras):
            t.columns[c].width = Inches(larg)
    for a in range(linhas):
        for b in range(colunas):
            celula = t.cell(a, b)
            celula.text = str(dados[a][b])
            celula.margin_left = celula.margin_right = Inches(0.06)
            celula.margin_top = celula.margin_bottom = Inches(0.03)
            celula.vertical_anchor = MSO_ANCHOR.MIDDLE
            par = celula.text_frame.paragraphs[0]
            par.alignment = PP_ALIGN.LEFT if b == 0 else PP_ALIGN.CENTER
            for r in par.runs:
                r.font.size, r.font.name = Pt(tam), FONTE
                r.font.bold = a == 0
                r.font.color.rgb = BRANCO if a == 0 else PRETO
            celula.fill.solid()
            celula.fill.fore_color.rgb = AZUL if a == 0 else (CLARO if a % 2 == 0 else BRANCO)
    return t


# ------------------------------------------------------------------ figura com as equações
def figura_equacoes():
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, eixo = plt.subplots(figsize=(7.2, 4.6))
    eixo.axis("off")
    blocos = [
        ("Objetivo regularizado do gradient boosting",
         r"$\mathcal{L}=\sum_{i=1}^{n}\ell\,(y_i,F(\mathbf{x}_i))+\sum_{m=1}^{M}\left(\gamma T+\frac{\lambda}{2}\sum_{j}w_j^2\right)$"),
        ("Calibração de Platt", r"$\hat{p}^{\,cal}_i=\sigma\left(a\,\ln\frac{s_i}{1-s_i}+b\right)$"),
        ("Revocação no topo da fila e lift", r"$R(c)=\frac{\sum_{i\in S_c}y_i}{\sum_{i}y_i}\qquad \mathrm{Lift}(c)=\frac{R(c)}{c}$"),
        ("Redução do CAC", r"$\Delta_{CAC}(c)=1-\frac{1}{\mathrm{Lift}(c)}$"),
        ("Valor de Shapley",
         r"$\phi_j=\sum_{S\subseteq F\setminus\{j\}}\frac{|S|!\,(|F|-|S|-1)!}{|F|!}\left[f_{S\cup\{j\}}(\mathbf{x})-f_S(\mathbf{x})\right]$"),
    ]
    for k, (rotulo, formula) in enumerate(blocos):
        y = 0.95 - k * 0.2
        eixo.text(0.0, y, rotulo, fontsize=10, color="#E07B39", fontweight="bold", va="top")
        eixo.text(0.02, y - 0.055, formula, fontsize=15, color="#222B36", va="top")
    destino = FIG / "slide_equacoes.png"
    fig.savefig(destino, dpi=220, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return destino


# ------------------------------------------------------------------ valores
m, ic = R["teste"]["metricas"], R["teste"]["ic95"]
val = R["comparacao_modelos_validacao"]
est = {(e["algoritmo"], e["estrategia"]): e for e in R["comparacao_estrategias_validacao"]}
pol = {round(c["capacidade"], 2): c for c in R["politica"]["cenarios"]}
cons = {round(c["capacidade"], 2): c for c in R["politica_conservadora"]}
abl, rt, decis = R["ablacao"], R["robustez_temporal"], R["decis_teste"]
shap_imp = R["shap_importancia"]
shap_macro = sum(shap_imp[v] for v in ("taxa_variacao_emprego", "indice_precos_consumidor",
                                         "indice_confianca_consumidor")) / sum(shap_imp.values())

# ------------------------------------------------------------------ 1. capa
contador[0] += 1
s = prs.slides.add_slide(EM_BRANCO)
retangulo(s, 0, 0, 13.333, 7.5, AZUL)
retangulo(s, 0, 5.05, 13.333, 0.05, LARANJA)
caixa(s, 1, 0.8, 11.3, 0.4, "CENTRO UNIVERSITÁRIO IESB", 16, BRANCO, True, PP_ALIGN.CENTER)
caixa(s, 1, 1.2, 11.3, 0.4, "Ciência de Dados e Inteligência Artificial · Projeto Integrador Aplicado IV",
      13, RGBColor(0xC9, 0xD8, 0xEA), False, PP_ALIGN.CENTER)
caixa(s, 1, 2.2, 11.3, 1.9, "Lead Scoring em Mídia Paga: Ciência de Dados Aplicada à Priorização Comercial "
      "e à Redução do Custo de Aquisição de Clientes", 32, BRANCO, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
caixa(s, 1, 4.3, 11.3, 0.4, "Avaliação P2 — Trabalho final: artigo, notebook e dashboard", 15,
      RGBColor(0xC9, 0xD8, 0xEA), False, PP_ALIGN.CENTER)
caixa(s, 1, 5.4, 11.3, 0.4, "Asafe Ferreira Gomes", 18, BRANCO, True, PP_ALIGN.CENTER)
caixa(s, 1, 5.85, 11.3, 0.4, "Orientação: Profa. Simone de Araújo Góes Assis", 13,
      RGBColor(0xC9, 0xD8, 0xEA), False, PP_ALIGN.CENTER)
caixa(s, 1, 6.5, 11.3, 0.4, "Brasília — 2026", 12, RGBColor(0xC9, 0xD8, 0xEA), False, PP_ALIGN.CENTER)

# ------------------------------------------------------------------ 2. problema
s = novo_slide("Problema", "A inteligência para na porta do time comercial")
topicos(s, [
    "Na mídia paga o anúncio captura um lead, que depois é trabalhado por uma equipe comercial.",
    "A captação já é otimizada por dados. O atendimento segue por ordem de chegada.",
    "O volume de leads cresce com a verba; a capacidade da equipe, não.",
    "Resultado: leads promissores esperam na fila e o CAC sobe.",
], 0.6, 1.7, 6.3, 3.2, 17)
etapas = [("Anúncio", "IA otimiza a entrega", AZUL), ("Lead", "Dado capturado", AZUL),
          ("Fila", "Ordem de chegada", VERMELHO), ("Venda", "Receita", VERDE)]
for k, (nome, sub, cor) in enumerate(etapas):
    x = 7.3 + k * 1.45
    retangulo(s, x, 2.3, 1.3, 0.95, cor, None, MSO_SHAPE.ROUNDED_RECTANGLE)
    caixa(s, x, 2.3, 1.3, 0.95, nome, 15, BRANCO, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    caixa(s, x - 0.1, 3.3, 1.5, 0.5, sub, 10, cor, True, PP_ALIGN.CENTER)
caixa(s, 10.1, 1.85, 1.5, 0.35, "A LACUNA ▼", 11, VERMELHO, True, PP_ALIGN.CENTER)
destaque(s, 0.6, 5.0, 12.1, 1.75, "Questão de pesquisa",
         "É possível estimar, com acurácia e calibração suficientes para uso operacional, a probabilidade de conversão "
         "de um lead a partir exclusivamente dos atributos disponíveis no momento de sua entrada, de modo a permitir a "
         "priorização do atendimento comercial e a consequente redução do custo de aquisição de clientes?", AZUL, 15)

# ------------------------------------------------------------------ 3. objetivos
s = novo_slide("Objetivos", "Da base tratada na P1 à solução completa na P2")
destaque(s, 0.6, 1.65, 12.1, 1.15, "Objetivo geral",
         "Desenvolver e validar uma solução de Ciência de Dados capaz de estimar a probabilidade de conversão de leads "
         "a partir dos atributos disponíveis na captação, para priorizar o atendimento e reduzir o CAC.", AZUL, 14)
caixa(s, 0.6, 3.0, 5.8, 0.35, "P1 — JÁ ENTREGUE", 12, CINZA, True)
topicos(s, ["Problema, questão de pesquisa e estado da arte", "Seleção da base e modelo relacional",
            "Análise exploratória com testes de associação", "Diagnóstico e tratamento de 7 anomalias",
            "Dicionário de dados com o papel de cada variável"], 0.6, 3.4, 5.8, 3.3, 14, CINZA)
caixa(s, 6.9, 3.0, 5.8, 0.35, "P2 — ESTA ENTREGA", 12, LARANJA, True)
topicos(s, ["Três modelos comparados: regressão logística, floresta aleatória e XGBoost",
            "Três estratégias para o desbalanceamento", "Calibração das probabilidades",
            "Interpretabilidade por valores de Shapley", "Política de priorização e impacto no CAC",
            "Robustez: ablação e partição temporal", "Notebook reproduzível e dashboard interativo"],
        6.9, 3.4, 5.8, 3.6, 14)

# ------------------------------------------------------------------ 4. referencial
s = novo_slide("Referencial teórico", "O que a literatura já resolveu e onde está a lacuna")
tabela(s, [
    ["Tema", "Referências", "Contribuição usada neste trabalho"],
    ["Conversão em marketing direto", "Moro, Cortez e Rita (2014); Monat (2011)",
     "Mesma base; alerta sobre o vazamento na duração da ligação"],
    ["Desbalanceamento", "Chawla et al. (2002); He e Garcia (2009)", "SMOTE e ponderação de classes"],
    ["Avaliação", "Saito e Rehmsmeier (2015); Davis e Goadrich (2006)",
     "Curva de precisão-revocação no lugar da acurácia"],
    ["Ensembles de árvores", "Breiman (2001); Friedman (2001); Chen e Guestrin (2016)",
     "Floresta aleatória e XGBoost"],
    ["Calibração", "Platt (1999); Zadrozny e Elkan (2002); Niculescu-Mizil e Caruana (2005)",
     "Score lido como probabilidade"],
    ["Interpretabilidade", "Lundberg e Lee (2017); Lundberg et al. (2020)", "Valores de Shapley; LGPD, art. 20"],
    ["Vazamento e desvio de conceito", "Kaufman et al. (2012); Gama et al. (2014)",
     "Exclusão de variáveis e validação temporal"],
], 0.6, 1.65, 12.1, 3.6, [3.0, 4.6, 4.5], 11)
destaque(s, 0.6, 5.5, 12.1, 1.3, "Lacuna explorada",
         "A literatura se concentra na discriminação. Este trabalho avança para a decisão: calibração, política sob "
         "restrição de capacidade e separação entre o sinal do lead e o sinal do contexto. As soluções de mercado "
         "atuam antes da captação (plataformas de anúncios) ou como caixas-pretas (CRMs).", LARANJA, 13)

# ------------------------------------------------------------------ 5. dados
s = novo_slide("Dados e preparação", "Base real e pública, tratada em sete etapas")
cartao(s, 0.6, 1.65, 2.85, 1.25, i(R["base"]["registros_brutos"]), "contatos reais de marketing direto (UCI, 2008-2010)", AZUL)
cartao(s, 3.68, 1.65, 2.85, 1.25, p(R["base"]["taxa_conversao"], 2), f"taxa de conversão ({n(R['base']['razao_desbalanceamento'], 2)} para 1)", LARANJA)
cartao(s, 6.76, 1.65, 2.85, 1.25, "99,97%", "dos registros preservados após o tratamento", VERDE)
cartao(s, 9.84, 1.65, 2.85, 1.25, str(R["base"]["n_preditoras"]), "variáveis preditoras, todas conhecidas antes da abordagem", AZUL)
tabela(s, [
    ["Anomalia", "Decisão"],
    ["12 linhas duplicadas", "Remoção"],
    [f"Ausência codificada como texto ({p(R['tratamento']['registros_com_unknown'] / R['base']['registros_tratados'])} dos registros)",
     "Categoria explícita e indicadores de ausência"],
    [f"Sentinela 999 em {p(R['tratamento']['sentinela_999'] / R['base']['registros_tratados'])} dos registros",
     "Indicador de contato prévio e imputação pela mediana"],
    ["Duração do contato (vazamento)", "Excluída do conjunto preditivo"],
    ["Valores extremos no número de contatos", "Limitação ao percentil 99"],
    ["Multicolinearidade macroeconômica (VIF de 64)", "Remoção de 2 variáveis; VIF máximo de 3,3"],
], 0.6, 3.15, 7.4, 3.3, [3.9, 3.5], 11)
destaque(s, 8.3, 3.15, 4.4, 3.3, "Ressalva metodológica",
         "A base descreve telemarketing bancário, e não mídia paga digital. Ela é usada como análogo estrutural: "
         "um contato é captado, uma origem é registrada, uma equipe aborda e um desfecho binário é observado.\n\n"
         "Transfere-se o método, e não os coeficientes.", LARANJA, 13)

# ------------------------------------------------------------------ 6. metodologia
s = novo_slide("Metodologia", "Três níveis de separação entre treino e avaliação")
passos = [("Tratamento", "7 etapas da P1"), ("Partição", "60 / 20 / 20\nestratificada"),
          ("Pipeline", "codificação e\npadronização"), ("Treino", "validação cruzada\nde 5 partes"),
          ("Validação", "estratégia,\ncalibração, limiar"), ("Teste", "usado uma\núnica vez")]
for k, (nome, sub) in enumerate(passos):
    x = 0.6 + k * 2.05
    retangulo(s, x, 1.75, 1.85, 0.75, AZUL if k < 5 else VERDE, None, MSO_SHAPE.CHEVRON if 0 < k else MSO_SHAPE.PENTAGON)
    caixa(s, x + 0.15, 1.75, 1.55, 0.75, nome, 13, BRANCO, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    caixa(s, x, 2.55, 1.85, 0.6, sub, 10, CINZA, False, PP_ALIGN.CENTER)
caixa(s, 0.6, 3.4, 6, 0.35, "MODELOS E ESTRATÉGIAS", 12, LARANJA, True)
topicos(s, ["Regressão logística ponderada como linha de base", "Floresta aleatória e XGBoost com busca aleatória de hiperparâmetros",
            "Desbalanceamento: ponderação de classes, SMOTE e ajuste do limiar",
            "Calibração: escala de Platt e regressão isotônica"], 0.6, 3.8, 6.0, 3, 14)
caixa(s, 6.9, 3.4, 6, 0.35, "MÉTRICAS E SUA JUSTIFICATIVA", 12, LARANJA, True)
topicos(s, ["AUC-PR: a acurácia engana, pois prever \"não converte\" para todos acerta 88,7%",
            "Revocação nos 20% do topo e lift: respondem ao uso real, que é ordenar uma fila",
            "Brier e ECE: medem se o score pode ser lido como probabilidade",
            "Intervalos de confiança por bootstrap (1.000 reamostragens)"], 6.9, 3.8, 5.8, 3, 14)

# ------------------------------------------------------------------ 7. matemática
s = novo_slide("Fundamentação matemática", "As formulações que sustentam a solução")
imagem(s, figura_equacoes(), 0.6, 1.65, h=5.2)
destaque(s, 8.6, 1.75, 4.1, 2.3, "Leitura",
         "O XGBoost soma árvores que corrigem o erro das anteriores, com penalização do número de folhas (γ) "
         "e do valor das folhas (λ).", AZUL, 13)
destaque(s, 8.6, 4.25, 4.1, 2.5, "Consequência prática",
         "Como o custo é o mesmo com ou sem priorização, a redução percentual do CAC depende só do lift do modelo, "
         "e não das premissas de custo.", LARANJA, 13)

# ------------------------------------------------------------------ 8. comparação
s = novo_slide("Resultados · validação", "Algoritmos empatam; a reamostragem distorce a probabilidade")
linhas = [["Modelo (validação)", "AUC-PR", "AUC-ROC", "Top 20%", "Brier"]]
for nome, v in val.items():
    linhas.append([nome, n(v["AUC-PR"]), n(v["AUC-ROC"]), p(v["Revocação top 20%"]), n(v["Brier"])])
tabela(s, linhas, 0.6, 1.7, 5.9, 1.7, [2.3, 0.9, 0.9, 0.9, 0.9], 12)
linhas = [["XGBoost + estratégia", "AUC-PR", "Prob. média prevista", "F1 (limiar ótimo)"]]
for estrategia in ("Sem tratamento", "Ponderação de classes", "SMOTE"):
    v = est[("XGBoost", estrategia)]
    linhas.append([estrategia, n(v["AUC-PR"]), p(v["Prob. média prevista"]), n(v["F1 (limiar ótimo)"])])
tabela(s, linhas, 6.8, 1.7, 5.9, 1.7, [2.3, 1.0, 1.4, 1.2], 12)
destaque(s, 0.6, 3.75, 5.9, 2.9, "Algoritmos",
         f"O XGBoost vence por margem pequena ({n(val['XGBoost']['AUC-PR'])} contra {n(val['Regressão logística']['AUC-PR'])} "
         "da regressão logística). O sinal da base é quase aditivo: a escolha do algoritmo importa menos do que a "
         "preparação dos dados.\n\nO Brier ruim da regressão logística vem da ponderação de classes, e não de ordenação pior.",
         AZUL, 13)
destaque(s, 6.8, 3.75, 5.9, 2.9, "Desbalanceamento",
         f"As três vias ordenam igual. Mas a taxa real é de 11,3%, e a ponderação prevê em média "
         f"{p(est[('XGBoost', 'Ponderação de classes')]['Prob. média prevista'])}; o SMOTE, "
         f"{p(est[('XGBoost', 'SMOTE')]['Prob. média prevista'])}.\n\nCom o limiar ajustado, o F1 é o mesmo. "
         "Escolha: XGBoost sem reamostragem, com ajuste do limiar.", LARANJA, 13)

# ------------------------------------------------------------------ 9. teste
s = novo_slide("Resultados · teste", "Ordena quatro vezes melhor que o acaso e está calibrado")
cartao(s, 0.6, 1.65, 2.85, 1.25, n(m["AUC-PR"]), f"AUC-PR (sem modelo: {n(R['teste']['prevalencia'])})\nIC 95%: {n(ic['AUC-PR'][0])} a {n(ic['AUC-PR'][1])}", AZUL)
cartao(s, 3.68, 1.65, 2.85, 1.25, n(m["AUC-ROC"]), f"AUC-ROC (sem modelo: 0,500)\nIC 95%: {n(ic['AUC-ROC'][0])} a {n(ic['AUC-ROC'][1])}", AZUL)
cartao(s, 6.76, 1.65, 2.85, 1.25, p(m["Revocação top 20%"]), "das conversões nos 20% do topo da fila", VERDE)
cartao(s, 9.84, 1.65, 2.85, 1.25, n(m["ECE"], 4), f"erro de calibração (ECE); antes: {n(m['ECE sem calibração'], 4)}", LARANJA)
imagem(s, FIG / "fig02_curvas_teste.png", 0.6, 3.1, w=12.1)
caixa(s, 0.6, 6.55, 12.1, 0.45,
      f"Conjunto de teste: {i(R['particoes']['teste']['registros'])} leads que não participaram de nenhuma decisão. "
      "Compatível com Moro, Cortez e Rita (2014), que relatam AUC de 0,80 nesta base.", 12, CINZA)

# ------------------------------------------------------------------ 10. ganho
s = novo_slide("Resultados · ganho acumulado", "A separação está concentrada no topo da fila")
imagem(s, FIG / "fig03_ganho_decis.png", 0.6, 1.65, w=8.3)
destaque(s, 9.2, 1.7, 3.5, 1.55, "1º decil", f"Converte {p(decis[0]['taxa_conversao'])}, contra 11,3% da média. "
         f"Lift de {n(m['Lift 1º decil'], 2)}.", VERDE, 13)
destaque(s, 9.2, 3.4, 3.5, 1.55, "Calibração por decil", f"No 1º decil o modelo previu {p(decis[0]['prob_media_prevista'])} "
         f"e a conversão foi de {p(decis[0]['taxa_conversao'])}.", AZUL, 13)
destaque(s, 9.2, 5.1, 3.5, 1.7, "Leitura operacional", "Do 3º decil em diante as taxas variam pouco. O score sugere "
         "uma política de dois níveis: prioridade imediata ao topo.", LARANJA, 13)
caixa(s, 0.6, 5.55, 8.3, 1.2,
      f"Regra binária (limiar de {n(R['teste']['limiar']['limiar'])}): aborda {p(R['teste']['limiar']['pct_leads_abordados'])} "
      f"dos leads, com precisão de {p(R['teste']['limiar']['precisao'])} e revocação de {p(R['teste']['limiar']['revocacao'])}.",
      13, CINZA)

# ------------------------------------------------------------------ 11. shap
s = novo_slide("Interpretabilidade", "O contexto macroeconômico domina o score")
imagem(s, FIG / "fig04_shap_global.png", 0.6, 1.65, w=8.6)
destaque(s, 9.5, 1.7, 3.2, 2.2, "O que pesa", f"Os três indicadores macroeconômicos somam {p(shap_macro, 0)} da importância. "
         "Depois vêm canal, mês e histórico de contato.", AZUL, 13)
destaque(s, 9.5, 4.1, 3.2, 2.7, "Por que importa", "Explicação individual por lead: a equipe entende a pontuação e "
         "atende-se ao direito de revisão de decisões automatizadas (LGPD, art. 20).", LARANJA, 13)
caixa(s, 0.6, 5.5, 8.6, 1.3, "Valores de Shapley exatos para árvores (TreeSHAP). A soma das contribuições com o valor de "
      "base reproduz as log-chances da predição, propriedade verificada no notebook.", 12, CINZA)

# ------------------------------------------------------------------ 12. robustez
s = novo_slide("Robustez", "Quanto do ganho vem do lead e quanto vem do contexto?")
linhas = [["Conjunto de variáveis", "AUC-ROC", "Top 20%", "AUC-ROC intracontexto"]]
for a in abl:
    linhas.append([a["Conjunto de variáveis"], n(a["AUC-ROC"]), p(a["Revocação top 20%"]), n(a["AUC-ROC intracontexto"])])
tabela(s, linhas, 0.6, 1.7, 7.4, 2.0, [3.9, 1.0, 1.0, 1.5], 12)
cartao(s, 8.3, 1.7, 2.1, 2.0, n(rt["AUC-ROC"]), "AUC-ROC em partição temporal (treino nos primeiros 80%)", VERMELHO)
cartao(s, 10.6, 1.7, 2.1, 2.0, f"{p(rt['taxa_conversao_treino'])} → {p(rt['taxa_conversao_teste'])}",
       "taxa de conversão: treino e período final", VERMELHO)
topicos(s, [
    "Leads do mesmo dia compartilham o mesmo contexto macroeconômico: essas variáveis não ordenam a fila daquele dia.",
    f"Entre leads contemporâneos, a AUC-ROC é de {n(abl[0]['AUC-ROC intracontexto'])}: a ordenação existe, mas é modesta.",
    "Em partição temporal o desempenho cai e a calibração se perde: é um caso de desvio de conceito.",
], 0.6, 4.0, 12.1, 1.7, 14)
destaque(s, 0.6, 5.65, 12.1, 1.15, "Conclusão", "O resultado do teste é um teto, válido para uma base acumulada de vários "
         "períodos. Para a fila do dia, a referência é a medida intracontexto. O modelo exige monitoramento e reciclagem.",
         LARANJA, 13)

# ------------------------------------------------------------------ 13. CAC
s = novo_slide("Política de priorização", "Com capacidade de 20%, o CAC cai entre "
               f"{p(cons[0.2]['reducao_cac_intracontexto_pct'] / 100, 0)} e {p(pol[0.2]['reducao_cac_pct'] / 100, 0)}")
imagem(s, FIG / "fig05_politica_cac.png", 0.6, 1.65, w=7.6)
linhas = [["Capacidade", "Redução do CAC (fila única)", "Redução do CAC (fila do dia)"]]
for c in (0.1, 0.2, 0.3, 0.4, 0.5):
    linhas.append([p(c, 0), p(cons[c]["reducao_cac_fila_unica_pct"] / 100), p(cons[c]["reducao_cac_intracontexto_pct"] / 100)])
tabela(s, linhas, 8.5, 1.7, 4.2, 2.6, [1.2, 1.5, 1.5], 12)
destaque(s, 8.5, 4.5, 4.2, 2.3, "Teto e piso", "Fila única: o modelo escolhe os melhores leads de todos os períodos. "
         "Fila do dia: a mesma capacidade dentro de cada contexto. O ganho real fica entre os dois.", LARANJA, 12)
caixa(s, 0.6, 5.05, 7.6, 1.8,
      f"Em fila única, com 20% de capacidade: {i(pol[0.2]['conversoes_modelo'])} conversões contra "
      f"{i(pol[0.2]['conversoes_ordem_chegada'])} por ordem de chegada, sem verba adicional. "
      f"Premissas ilustrativas: R$ {n(R['politica']['cpl'], 0)} por lead e R$ {n(R['politica']['custo_abordagem'], 0)} "
      "por abordagem. A redução percentual não depende delas.", 13, CINZA)

# ------------------------------------------------------------------ 14. dashboard
s = novo_slide("Dashboard interativo", "Demonstração: do indicador à decisão")
imagem(s, FIG / "telas" / "tela_3d.png", 0.6, 1.65, h=5.2)
caixa(s, 7.6, 1.65, 5.1, 0.35, "ROTEIRO DA DEMONSTRAÇÃO", 12, LARANJA, True)
topicos(s, [
    "Visão executiva: filtros e conversão observada contra prevista",
    "Explorador 3D: mapa dos leads, superfície de resposta e superfície do CAC",
    "Simulador: capacidade e custos; fila única ou fila do dia",
    "Pontuar um lead: score e explicação recalculados na hora",
    "Fila priorizada e assistente de IA em linguagem natural",
], 7.6, 2.05, 5.1, 3.6, 13)
destaque(s, 7.6, 5.55, 5.1, 1.25, "Consistência", "O notebook grava as métricas e o modelo; o dashboard e o artigo leem "
         "os mesmos arquivos. Os três produtos mostram os mesmos números.", AZUL, 12)

# ------------------------------------------------------------------ 15. conclusões
s = novo_slide("Conclusões", "Resposta à questão de pesquisa: sim, com ressalva")
caixa(s, 0.6, 1.65, 6, 0.35, "CONTRIBUIÇÕES", 12, VERDE, True)
topicos(s, [
    f"O score ordena e está calibrado: AUC-PR de {n(m['AUC-PR'])} contra {n(R['teste']['prevalencia'])}; ECE de {n(m['ECE'], 4)}",
    "Para score de fila, treinar sem reamostragem e ajustar o limiar é melhor que ponderar ou aplicar SMOTE",
    "A separação entre sinal do lead e sinal do contexto muda a estimativa de impacto no CAC",
    "Notebook reproduzível e dashboard com simulador e explicação individual",
], 0.6, 2.05, 5.9, 4.0, 14)
caixa(s, 6.9, 1.65, 6, 0.35, "LIMITAÇÕES", 12, VERMELHO, True)
topicos(s, [
    "Base de telemarketing bancário de 2008 a 2010: transfere-se o método",
    "Desempenho cai em partição temporal; período inclui a crise de 2008",
    "Custos da simulação são premissas; dados observacionais não medem efeito causal",
    "Equidade do score entre grupos não foi avaliada",
], 6.9, 2.05, 5.8, 4.0, 14)
destaque(s, 0.6, 5.5, 12.1, 1.3, "Continuidade", "Aplicar a uma base proprietária de mídia paga com atributos de comportamento "
         "digital; validação temporal deslizante e reciclagem programada; avaliação de equidade; experimento controlado "
         "para medir o efeito causal da priorização.", AZUL, 13)

# ------------------------------------------------------------------ 16. referências
s = novo_slide("Referências principais", "Referências e declaração de uso de IA")
refs = [
    "BREIMAN, L. Random forests. Machine Learning, v. 45, n. 1, 2001.",
    "CHAWLA, N. V. et al. SMOTE: synthetic minority over-sampling technique. JAIR, v. 16, 2002.",
    "CHEN, T.; GUESTRIN, C. XGBoost: a scalable tree boosting system. KDD, 2016.",
    "GAMA, J. et al. A survey on concept drift adaptation. ACM Computing Surveys, v. 46, n. 4, 2014.",
    "KAUFMAN, S. et al. Leakage in data mining. ACM TKDD, v. 6, n. 4, 2012.",
    "LUNDBERG, S. M.; LEE, S.-I. A unified approach to interpreting model predictions. NeurIPS, 2017.",
    "MORO, S.; CORTEZ, P.; RITA, P. A data-driven approach to predict the success of bank telemarketing. "
    "Decision Support Systems, v. 62, 2014.",
    "NICULESCU-MIZIL, A.; CARUANA, R. Predicting good probabilities with supervised learning. ICML, 2005.",
    "PLATT, J. C. Probabilistic outputs for support vector machines. Advances in Large Margin Classifiers, 1999.",
    "SAITO, T.; REHMSMEIER, M. The precision-recall plot is more informative than the ROC plot. PLoS ONE, 2015.",
]
topicos(s, refs, 0.6, 1.65, 12.1, 4.2, 12)
destaque(s, 0.6, 5.75, 12.1, 1.05, "Uso de inteligência artificial",
         "Ferramenta de IA generativa (Claude, da Anthropic) usada como apoio na programação, na redação e na revisão. "
         "Decisões metodológicas, conferência dos resultados e conteúdo final são de responsabilidade do autor. "
         "A lista completa de referências está no artigo.", CINZA, 12)

assert contador[0] == TOTAL, contador[0]
destino = RAIZ / "04_Apresentacao" / "Apresentacao_Lead_Scoring_P2.pptx"
prs.save(str(destino))
print("Apresentação gravada:", destino, f"({contador[0]} slides)")
