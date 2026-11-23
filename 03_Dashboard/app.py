"""Dashboard interativo do projeto Lead Scoring em Mídia Paga (P2 - Projeto Integrador IV - IESB).

Execução local:  streamlit run 03_Dashboard/app.py
Os dados vêm da pasta artefatos/, gerada pelo notebook 02_Notebook/Lead_Scoring_P2.ipynb.

Assistente de IA: usa a API do Claude quando há uma chave (variável ANTHROPIC_API_KEY ou campo
na barra lateral). Sem chave, responde em modo local, com regras sobre os mesmos dados.
"""
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb
from sklearn.decomposition import PCA

RAIZ = Path(__file__).resolve().parent.parent
ARTEFATOS = RAIZ / "artefatos"

# Paleta
FUNDO, PAINEL, BORDA = "#0A0F1E", "#121A2F", "#243049"
TEXTO, SUAVE = "#E8ECF5", "#8C97B2"
CIANO, VIOLETA, VERDE, VERMELHO, AMBAR, CINZA = "#22D3EE", "#8B5CF6", "#34D399", "#F87171", "#FBBF24", "#5B6785"
ESCALA = [[0, "#1E2A4A"], [0.25, "#3B4BA8"], [0.5, "#8B5CF6"], [0.75, "#22D3EE"], [1, "#34D399"]]

ROTULOS = {
    "profissao": "Profissão", "estado_civil": "Estado civil", "escolaridade": "Escolaridade",
    "inadimplencia": "Inadimplência", "canal_contato": "Canal de contato", "mes_contato": "Mês do contato",
    "resultado_campanha_anterior": "Resultado da campanha anterior", "faixa_idade": "Faixa etária",
    "idade": "Idade", "n_contatos_campanha": "Contatos na campanha atual",
    "dias_desde_ultimo_contato": "Dias desde o último contato",
    "n_contatos_anteriores": "Contatos em campanhas anteriores",
    "taxa_variacao_emprego": "Taxa de variação do emprego",
    "indice_precos_consumidor": "Índice de preços ao consumidor",
    "indice_confianca_consumidor": "Índice de confiança do consumidor",
    "contato_previo": "Já teve contato prévio", "flag_escolaridade_desconhecida": "Escolaridade não informada",
    "dia_semana": "Dia da semana", "prob_conversao": "Probabilidade de conversão",
}
ORDEM_MESES = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
ORDEM_FAIXA = ["ate 25", "26-35", "36-45", "46-55", "56-65", "66+"]
CHAVE_CONTEXTO = ["mes_contato", "taxa_variacao_emprego", "indice_precos_consumidor",
                  "indice_confianca_consumidor"]
MODELO_IA = "claude-opus-5-5"

st.set_page_config(page_title="Lead Scoring em Mídia Paga", page_icon="◆", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Inter', 'Segoe UI', sans-serif; }}
.stApp {{ background: radial-gradient(1200px 600px at 85% -10%, rgba(139,92,246,.18), transparent 60%),
                     radial-gradient(900px 500px at -10% 10%, rgba(34,211,238,.12), transparent 55%), {FUNDO}; }}
.block-container {{ padding-top: 1.6rem; max-width: 1500px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0D1428 0%, #0A0F1E 100%);
                                    border-right: 1px solid {BORDA}; }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}
.hero {{ border: 1px solid {BORDA}; border-radius: 20px; padding: 26px 30px; margin-bottom: 18px;
         background: linear-gradient(135deg, rgba(34,211,238,.10), rgba(139,92,246,.14) 60%, rgba(18,26,47,.6)); }}
.hero .selo {{ display: inline-block; font-size: .72rem; font-weight: 700; letter-spacing: .14em; color: {CIANO};
               border: 1px solid rgba(34,211,238,.4); border-radius: 999px; padding: 4px 12px; margin-bottom: 12px; }}
.hero h1 {{ margin: 0; font-size: 2.3rem; font-weight: 800; padding: 0;
            background: linear-gradient(90deg, #FFFFFF 0%, {CIANO} 55%, {VIOLETA} 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.hero p {{ margin: 8px 0 0 0; color: {SUAVE}; font-size: 1.02rem; max-width: 900px; }}
div[data-testid="stMetric"] {{ background: linear-gradient(160deg, rgba(18,26,47,.95), rgba(18,26,47,.6));
    border: 1px solid {BORDA}; border-radius: 16px; padding: 16px 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,.25); transition: transform .15s ease, border-color .15s ease; }}
div[data-testid="stMetric"]:hover {{ transform: translateY(-2px); border-color: rgba(34,211,238,.55); }}
div[data-testid="stMetricLabel"] p {{ color: {SUAVE}; font-size: .8rem; font-weight: 600; letter-spacing: .03em; }}
div[data-testid="stMetricValue"] {{ font-weight: 800; font-size: 1.9rem; color: #FFFFFF; }}
div[data-testid="stExpander"] {{ border: 1px solid {BORDA}; border-radius: 14px; background: rgba(18,26,47,.55); }}
button[data-baseweb="tab"] {{ font-weight: 600; }}
div[data-baseweb="tab-highlight"] {{ background: linear-gradient(90deg, {CIANO}, {VIOLETA}); height: 3px; }}
.stButton > button, .stDownloadButton > button {{ border-radius: 12px; border: 1px solid {BORDA}; font-weight: 600; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color: {CIANO}; color: {CIANO}; }}
div[data-testid="stPlotlyChart"] {{ border: 1px solid {BORDA}; border-radius: 16px; background: rgba(18,26,47,.5);
                                    padding: 6px; }}
div[data-testid="stDataFrame"] {{ border: 1px solid {BORDA}; border-radius: 14px; overflow: hidden; }}
.nota {{ border-left: 3px solid {VIOLETA}; background: rgba(139,92,246,.10); border-radius: 0 12px 12px 0;
         padding: 12px 16px; color: {TEXTO}; font-size: .95rem; margin: 6px 0 14px 0; }}
.alerta {{ border-left: 3px solid {AMBAR}; background: rgba(251,191,36,.09); border-radius: 0 12px 12px 0;
           padding: 12px 16px; color: {TEXTO}; font-size: .95rem; margin: 10px 0; }}
.cartao-ia {{ border: 1px solid {BORDA}; border-radius: 14px; padding: 12px 16px; background: rgba(18,26,47,.7);
              color: {SUAVE}; font-size: .88rem; }}
.pilula {{ display: inline-block; border-radius: 999px; padding: 3px 12px; font-size: .75rem; font-weight: 700; }}
.pilula.ok {{ background: rgba(52,211,153,.15); color: {VERDE}; border: 1px solid rgba(52,211,153,.4); }}
.pilula.local {{ background: rgba(251,191,36,.12); color: {AMBAR}; border: 1px solid rgba(251,191,36,.4); }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- carga
@st.cache_resource
def carregar_modelo():
    return joblib.load(ARTEFATOS / "modelo_lead_scoring.joblib")


@st.cache_data
def carregar_dados():
    base_ = pd.read_csv(ARTEFATOS / "base_pontuada.csv")
    resultados = json.loads((ARTEFATOS / "resultados.json").read_text(encoding="utf-8"))
    shap_ = pd.read_csv(ARTEFATOS / "shap_teste.csv", index_col=0)
    return base_, resultados, shap_


if not (ARTEFATOS / "modelo_lead_scoring.joblib").exists():
    st.error("Artefatos não encontrados. Execute primeiro o notebook 02_Notebook/Lead_Scoring_P2.ipynb.")
    st.stop()

modelo = carregar_modelo()
base, res, shap_teste = carregar_dados()
teste = base[base["particao"] == "teste"].copy()
PREDITORAS, NOMINAIS = modelo["preditoras"], modelo["nominais"]
TAXA_GERAL = float(base["converteu"].mean())


# ----------------------------------------------------------------------------- funções
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def pontuar(dados):
    """Score bruto (ordena a fila) e probabilidade calibrada."""
    bruto = modelo["pipeline"].predict_proba(dados[PREDITORAS])[:, 1]
    if modelo["metodo_calibracao"] == "isotonica":
        return bruto, modelo["calibrador"].predict(bruto)
    return bruto, modelo["calibrador"].predict_proba(logit(bruto).reshape(-1, 1))[:, 1]


def contribuicoes_shap(dados):
    """Valores de Shapley (TreeSHAP exato do XGBoost), somados por variável original."""
    pre, clf = modelo["pipeline"].named_steps["pre"], modelo["pipeline"].named_steps["clf"]
    colunas = list(pre.get_feature_names_out())
    origem = [next((n for n in NOMINAIS if c.startswith(n + "_")), c) for c in colunas]
    matriz = xgb.DMatrix(pre.transform(dados[PREDITORAS]), feature_names=colunas)
    contrib = clf.get_booster().predict(matriz, pred_contribs=True)
    por_variavel = pd.DataFrame(contrib[:, :-1], columns=colunas).T.groupby(origem).sum().T
    return por_variavel, contrib[:, -1]


def montar_leads(atributos):
    """Recebe um dicionário (valores escalares ou vetores) e devolve a tabela pronta para o modelo."""
    df = pd.DataFrame(atributos)
    df["faixa_idade"] = pd.cut(df["idade"], bins=[0, 25, 35, 45, 55, 65, 200], labels=ORDEM_FAIXA).astype(str)
    df["flag_escolaridade_desconhecida"] = (df["escolaridade"] == "desconhecido").astype(int)
    df["dias_desde_ultimo_contato"] = np.where(df["contato_previo"] == 1, df["dias_desde_ultimo_contato"],
                                               float(modelo["mediana_pdays"]))
    return df


def revocacao_no_topo(y, score, fracao):
    y = np.asarray(y)
    if y.sum() == 0:
        return 0.0
    ordem = np.argsort(-np.asarray(score), kind="stable")
    return float(y[ordem][: int(np.ceil(fracao * len(y)))].sum() / y.sum())


def revocacao_intracontexto(dados, fracao):
    """Conversões capturadas quando a fração superior é atendida dentro de cada contexto."""
    total = dados["converteu"].sum()
    if total == 0:
        return 0.0
    capturadas = 0
    for _, grupo in dados.groupby(CHAVE_CONTEXTO):
        k = int(np.ceil(fracao * len(grupo)))
        capturadas += grupo.sort_values("score_bruto", ascending=False, kind="stable")["converteu"].iloc[:k].sum()
    return float(capturadas / total)


def brl(valor):
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def inteiro(valor):
    return f"{valor:,.0f}".replace(",", ".")


def pct(valor, casas=1):
    return f"{valor * 100:.{casas}f}%".replace(".", ",")


def dec(valor, casas=3):
    return f"{valor:.{casas}f}".replace(".", ",")


def layout(fig, altura=380, titulo=None):
    fig.update_layout(
        height=altura, title=dict(text=titulo, font=dict(size=15, color=TEXTO)) if titulo else None,
        margin=dict(l=10, r=10, t=52 if titulo else 18, b=10), template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=TEXTO, size=12),
        legend=dict(orientation="h", y=-0.3, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=PAINEL, bordercolor=BORDA, font=dict(color=TEXTO)))
    fig.update_xaxes(showgrid=True, gridcolor="#1C2740", zerolinecolor="#2A3654")
    fig.update_yaxes(showgrid=True, gridcolor="#1C2740", zerolinecolor="#2A3654")
    return fig


def cena_3d(fig, x, y, z, altura=640, titulo=None):
    eixo = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#26324F", zerolinecolor="#33405F",
                showbackground=True, color=SUAVE)
    fig.update_layout(
        height=altura, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text=titulo, font=dict(size=15, color=TEXTO)) if titulo else None,
        margin=dict(l=0, r=50, t=46 if titulo else 6, b=0),
        font=dict(family="Inter, Segoe UI, sans-serif", color=TEXTO, size=11),
        scene=dict(xaxis=dict(title=x, **eixo), yaxis=dict(title=y, **eixo), zaxis=dict(title=z, **eixo),
                   camera=dict(eye=dict(x=1.55, y=1.55, z=0.95)), aspectmode="cube"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=0.02),
        hoverlabel=dict(bgcolor=PAINEL, bordercolor=BORDA, font=dict(color=TEXTO)))
    return fig


def hero(selo, titulo, texto):
    st.markdown(f'<div class="hero"><span class="selo">{selo}</span><h1>{titulo}</h1><p>{texto}</p></div>',
                unsafe_allow_html=True)


def nota(texto, classe="nota"):
    st.markdown(f'<div class="{classe}">{texto}</div>', unsafe_allow_html=True)


@st.cache_data
def mapa_explicacoes():
    """Projeção 3D (PCA) dos valores de Shapley do teste: leads próximos têm scores com os mesmos motivos."""
    valores = shap_teste.drop(columns="valor_base")
    pca = PCA(n_components=3, random_state=42).fit(valores)
    coordenadas = pd.DataFrame(pca.transform(valores), index=valores.index, columns=["c1", "c2", "c3"])
    principal = valores.abs().idxmax(axis=1).rename("motivo_principal")
    return coordenadas.join(principal), pca.explained_variance_ratio_


# ----------------------------------------------------------------------------- navegação
PAGINAS = ["1. Visão executiva", "2. Explorador 3D", "3. Desempenho do modelo", "4. Simulador de priorização",
           "5. Pontuar um lead", "6. Fila priorizada", "7. Assistente de IA"]
st.sidebar.markdown(f"<div style='font-size:1.25rem;font-weight:800;line-height:1.2'>◆ Lead Scoring"
                    f"<br><span style='color:{CIANO}'>em Mídia Paga</span></div>", unsafe_allow_html=True)
st.sidebar.caption("P2 · Projeto Integrador IV · IESB · Asafe Ferreira Gomes")
pagina = st.sidebar.radio("Seção", PAGINAS)
st.sidebar.divider()
st.sidebar.caption(
    f"Base: Bank Marketing (UCI), {inteiro(res['base']['registros_tratados'])} leads. "
    f"Modelo: {res['modelo_final']['algoritmo']} com calibração {res['modelo_final']['calibracao']}. "
    "Os números de desempenho usam apenas o conjunto de teste (20% da base).")


# ============================================================================= 1. VISÃO EXECUTIVA
if pagina.startswith("1"):
    m = res["teste"]["metricas"]
    hero("PAINEL EXECUTIVO", "Quem atender primeiro?",
         "O modelo estima a probabilidade de conversão de cada lead no momento em que ele entra. "
         "Filtre a base e veja onde está a conversão e o quanto o score a antecipa.")

    with st.expander("Filtros", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        f_prof = c1.multiselect("Profissão", sorted(base["profissao"].unique()), placeholder="Todos")
        f_canal = c2.multiselect("Canal de contato", sorted(base["canal_contato"].unique()), placeholder="Todos")
        f_mes = c3.multiselect("Mês do contato", [x for x in ORDEM_MESES if x in base["mes_contato"].unique()],
                               placeholder="Todos")
        f_hist = c4.multiselect("Resultado da campanha anterior",
                                sorted(base["resultado_campanha_anterior"].unique()), placeholder="Todos")
        c5, c6 = st.columns([2, 2])
        f_idade = c5.slider("Idade", int(base["idade"].min()), int(base["idade"].max()),
                            (int(base["idade"].min()), int(base["idade"].max())))
        f_particao = c6.radio("Registros", ["Base completa", "Somente conjunto de teste"], horizontal=True)

    recorte = base if f_particao == "Base completa" else teste
    recorte = recorte[recorte["idade"].between(*f_idade)]
    for coluna, selecao in [("profissao", f_prof), ("canal_contato", f_canal), ("mes_contato", f_mes),
                            ("resultado_campanha_anterior", f_hist)]:
        if selecao:
            recorte = recorte[recorte[coluna].isin(selecao)]
    if recorte.empty:
        st.warning("Nenhum lead atende aos filtros selecionados.")
        st.stop()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Leads no recorte", inteiro(len(recorte)),
              f"{len(recorte) / len(base):.1%} da base".replace(".", ","), delta_color="off")
    k2.metric("Conversões", inteiro(recorte["converteu"].sum()))
    k3.metric("Taxa de conversão", pct(recorte["converteu"].mean()),
              f"{(recorte['converteu'].mean() - TAXA_GERAL) * 100:+.1f}".replace(".", ",") + " p.p. vs. média")
    k4.metric("Prob. média prevista", pct(recorte["prob_conversao"].mean()))
    k5.metric("Captura no top 20%", pct(m["Revocação top 20%"]), "das conversões", delta_color="off")

    esquerda, direita = st.columns([1.15, 1])
    with esquerda:
        dimensao = st.selectbox("Comparar conversão por", ["profissao", "resultado_campanha_anterior", "mes_contato",
                                                           "canal_contato", "faixa_idade", "escolaridade",
                                                           "estado_civil", "inadimplencia", "dia_semana"],
                                format_func=lambda c: ROTULOS.get(c, c))
        tabela = recorte.groupby(dimensao).agg(leads=("converteu", "size"), conversoes=("converteu", "sum"),
                                               taxa=("converteu", "mean"),
                                               prob_prevista=("prob_conversao", "mean")).reset_index()
        ordem = {"mes_contato": ORDEM_MESES, "faixa_idade": ORDEM_FAIXA}.get(dimensao)
        if ordem:
            tabela = tabela.set_index(dimensao).reindex([o for o in ordem if o in tabela[dimensao].values]).reset_index()
        else:
            tabela = tabela.sort_values("taxa")
        fig = go.Figure()
        fig.add_bar(y=tabela[dimensao], x=tabela["taxa"] * 100, orientation="h", name="Conversão observada",
                    marker=dict(color=tabela["taxa"], colorscale=ESCALA, line=dict(width=0)),
                    customdata=tabela[["leads", "conversoes"]],
                    hovertemplate="%{y}<br>Conversão: %{x:.1f}%<br>Leads: %{customdata[0]}"
                                  "<br>Conversões: %{customdata[1]}<extra></extra>")
        fig.add_scatter(y=tabela[dimensao], x=tabela["prob_prevista"] * 100, mode="markers",
                        name="Probabilidade prevista pelo modelo",
                        marker=dict(color="#FFFFFF", size=10, symbol="diamond", line=dict(color=VIOLETA, width=2)))
        fig.add_vline(x=TAXA_GERAL * 100, line_dash="dash", line_color=AMBAR,
                      annotation_text="média geral", annotation_position="top")
        fig.update_xaxes(title="Taxa de conversão (%)")
        st.plotly_chart(layout(fig, 450, f"Conversão por {ROTULOS.get(dimensao, dimensao).lower()}"), width="stretch")
    with direita:
        visao = st.selectbox("Visão complementar", ["Mapa de calor: mês x canal", "Distribuição do score",
                                                    "Composição do recorte"])
        if visao.startswith("Mapa"):
            calor = recorte.pivot_table(index="canal_contato", columns="mes_contato", values="converteu",
                                        aggfunc="mean") * 100
            calor = calor[[x for x in ORDEM_MESES if x in calor.columns]]
            fig = go.Figure(go.Heatmap(z=calor.values, x=calor.columns, y=calor.index, colorscale=ESCALA,
                                       text=np.round(calor.values, 0), texttemplate="%{text:.0f}%",
                                       colorbar=dict(title="%"),
                                       hovertemplate="%{y} · %{x}<br>Conversão: %{z:.1f}%<extra></extra>"))
            st.plotly_chart(layout(fig, 450, "Taxa de conversão por mês e canal"), width="stretch")
        elif visao.startswith("Distribuição"):
            dist = recorte.assign(desfecho=recorte["converteu"].map({0: "Não converteu", 1: "Converteu"}))
            fig = px.histogram(dist, x="prob_conversao", color="desfecho", nbins=40, barmode="overlay",
                               histnorm="percent", opacity=0.75,
                               color_discrete_map={"Não converteu": CINZA, "Converteu": VERDE},
                               labels={"prob_conversao": "Probabilidade de conversão prevista", "desfecho": ""})
            fig.update_yaxes(title="% dos leads de cada grupo")
            st.plotly_chart(layout(fig, 450, "O score separa convertidos de não convertidos?"), width="stretch")
        else:
            arvore = recorte.groupby(["canal_contato", "profissao"]).agg(
                leads=("converteu", "size"), taxa=("converteu", "mean")).reset_index()
            fig = px.treemap(arvore, path=["canal_contato", "profissao"], values="leads", color="taxa",
                             color_continuous_scale=ESCALA,
                             labels={"taxa": "Conversão", "leads": "Leads"})
            fig.update_traces(hovertemplate="%{label}<br>Leads: %{value}<br>Conversão: %{color:.1%}<extra></extra>")
            st.plotly_chart(layout(fig, 450, "Tamanho = leads; cor = taxa de conversão"), width="stretch")

    with st.expander("Tabela do recorte"):
        exibir = tabela.rename(columns={dimensao: ROTULOS.get(dimensao, dimensao), "leads": "Leads",
                                        "conversoes": "Conversões", "taxa": "Taxa de conversão",
                                        "prob_prevista": "Probabilidade prevista"})
        st.dataframe(exibir.style.format({"Taxa de conversão": "{:.1%}", "Probabilidade prevista": "{:.1%}"}),
                     width="stretch", hide_index=True)


# ============================================================================= 2. EXPLORADOR 3D
elif pagina.startswith("2"):
    hero("EXPLORADOR 3D", "O modelo em três dimensões",
         "Gire, aproxime e passe o mouse. O mapa mostra como a IA agrupa os leads; as superfícies mostram como a "
         "probabilidade e o custo de aquisição respondem às variáveis.")
    aba_mapa, aba_resposta, aba_cac = st.tabs(["Mapa 3D dos leads", "Superfície de resposta do modelo",
                                               "Superfície do CAC"])

    # ---- mapa
    with aba_mapa:
        c1, c2, c3, c4 = st.columns([1.3, 1.2, 1, 1])
        modo = c1.radio("Eixos", ["Mapa de explicações da IA", "Variáveis à escolha"], horizontal=False)
        cor_por = c2.selectbox("Colorir por", ["Probabilidade de conversão", "Desfecho real",
                                               "Motivo principal do score", "Canal de contato", "Profissão"])
        amostra = c3.slider("Leads exibidos", 500, len(teste), 3000, step=500)
        capacidade_3d = c4.slider("Destacar o topo da fila (%)", 0, 100, 0, step=5,
                                  help="Mantém em destaque os leads que a equipe atenderia com essa capacidade.")

        coordenadas, variancia = mapa_explicacoes()
        dados = teste.join(coordenadas)
        corte_fila = dados["score_bruto"].quantile(1 - capacidade_3d / 100) if capacidade_3d else None
        dados = dados.sample(min(amostra, len(dados)), random_state=42)
        if modo.startswith("Mapa"):
            ex, ey, ez = "c1", "c2", "c3"
            nx, ny, nz = [f"Componente {k + 1} ({v:.0%})" for k, v in enumerate(variancia)]
        else:
            numericas = ["idade", "n_contatos_campanha", "n_contatos_anteriores", "taxa_variacao_emprego",
                         "indice_precos_consumidor", "indice_confianca_consumidor", "prob_conversao"]
            e1, e2, e3 = st.columns(3)
            ex = e1.selectbox("Eixo X", numericas, index=0, format_func=lambda c: ROTULOS.get(c, c))
            ey = e2.selectbox("Eixo Y", numericas, index=3, format_func=lambda c: ROTULOS.get(c, c))
            ez = e3.selectbox("Eixo Z", numericas, index=6, format_func=lambda c: ROTULOS.get(c, c))
            nx, ny, nz = ROTULOS.get(ex, ex), ROTULOS.get(ey, ey), ROTULOS.get(ez, ez)

        info = np.stack([dados["id_lead"], dados["prob_conversao"] * 100, dados["profissao"], dados["idade"],
                         dados["canal_contato"], dados["mes_contato"],
                         dados["converteu"].map({0: "não", 1: "sim"}),
                         dados["motivo_principal"].map(lambda v: ROTULOS.get(v, v))], axis=-1)
        dica = ("Lead %{customdata[0]}<br>Probabilidade: %{customdata[1]:.1f}%<br>Profissão: %{customdata[2]}"
                "<br>Idade: %{customdata[3]}<br>Canal: %{customdata[4]} · Mês: %{customdata[5]}"
                "<br>Converteu: %{customdata[6]}<br>Motivo principal: %{customdata[7]}<extra></extra>")
        opacidade = np.where(dados["score_bruto"] >= corte_fila, 0.95, 0.06) if corte_fila is not None else None
        fig = go.Figure()
        if cor_por == "Probabilidade de conversão":
            selecionados = dados if corte_fila is None else dados[dados["score_bruto"] >= corte_fila]
            if corte_fila is not None:
                fora = dados[dados["score_bruto"] < corte_fila]
                fig.add_scatter3d(x=fora[ex], y=fora[ey], z=fora[ez], mode="markers", name="Fora da capacidade",
                                  marker=dict(size=2.2, color="#3A4563", opacity=0.25), hoverinfo="skip")
            fig.add_scatter3d(x=selecionados[ex], y=selecionados[ey], z=selecionados[ez], mode="markers",
                              name="Leads", customdata=info[dados.index.isin(selecionados.index)], hovertemplate=dica,
                              marker=dict(size=3.8, color=selecionados["prob_conversao"], colorscale=ESCALA,
                                          cmin=0, cmax=0.8, opacity=0.9,
                                          colorbar=dict(title="Prob.", tickformat=".0%", len=0.6, x=0.98)))
        else:
            coluna = {"Desfecho real": "converteu", "Motivo principal do score": "motivo_principal",
                      "Canal de contato": "canal_contato", "Profissão": "profissao"}[cor_por]
            paleta = [CIANO, VIOLETA, VERDE, AMBAR, VERMELHO, "#60A5FA", "#F472B6", "#A3E635", "#FB923C",
                      "#E879F9", "#2DD4BF", "#CBD5E1"]
            for k, (valor, grupo) in enumerate(dados.groupby(coluna)):
                if corte_fila is not None:
                    grupo = grupo[grupo["score_bruto"] >= corte_fila]
                if grupo.empty:
                    continue
                nome = {0: "Não converteu", 1: "Converteu"}.get(valor, ROTULOS.get(valor, str(valor))) \
                    if coluna in ("converteu", "motivo_principal") else str(valor)
                cor = {0: "#4B5878", 1: VERDE}.get(valor, paleta[k % len(paleta)]) if coluna == "converteu" \
                    else paleta[k % len(paleta)]
                fig.add_scatter3d(x=grupo[ex], y=grupo[ey], z=grupo[ez], mode="markers", name=nome,
                                  customdata=info[dados.index.isin(grupo.index)], hovertemplate=dica,
                                  marker=dict(size=3.2, color=cor, opacity=0.85))
        st.plotly_chart(cena_3d(fig, nx, ny, nz, 660), width="stretch")
        if modo.startswith("Mapa"):
            nota(f"<b>Como ler.</b> Cada ponto é um lead do conjunto de teste. A posição vem da projeção em três "
                 f"componentes principais dos valores de Shapley, que explicam {pct(float(variancia.sum()), 0)} da "
                 "variação das explicações. Leads próximos receberam o score pelos mesmos motivos, e os grupos "
                 "separados correspondem, em grande parte, a contextos macroeconômicos diferentes.")
        if capacidade_3d:
            atendidos = teste[teste["score_bruto"] >= teste["score_bruto"].quantile(1 - capacidade_3d / 100)]
            a1, a2, a3 = st.columns(3)
            a1.metric("Leads em destaque", inteiro(len(atendidos)), f"{capacidade_3d}% da fila", delta_color="off")
            a2.metric("Conversão no grupo destacado", pct(atendidos["converteu"].mean()),
                      f"{atendidos['converteu'].mean() / teste['converteu'].mean():.1f}x a média".replace(".", ",", 1),
                      delta_color="off")
            a3.metric("Conversões capturadas", pct(atendidos["converteu"].sum() / teste["converteu"].sum()),
                      "do total do teste", delta_color="off")

    # ---- superfície de resposta
    with aba_resposta:
        variaveis = ["idade", "n_contatos_campanha", "taxa_variacao_emprego", "indice_precos_consumidor",
                     "indice_confianca_consumidor", "n_contatos_anteriores"]
        c1, c2, c3, c4 = st.columns(4)
        vx = c1.selectbox("Variável no eixo X", variaveis, index=0, format_func=lambda c: ROTULOS.get(c, c))
        vy = c2.selectbox("Variável no eixo Y", [v for v in variaveis if v != vx], index=1,
                          format_func=lambda c: ROTULOS.get(c, c))
        canal_s = c3.selectbox("Canal de contato", sorted(base["canal_contato"].unique()))
        hist_s = c4.selectbox("Resultado da campanha anterior", ["nonexistent", "failure", "success"])
        c5, c6, c7 = st.columns(3)
        prof_s = c5.selectbox("Profissão", sorted(base["profissao"].unique()), index=0)
        mes_s = c6.selectbox("Mês do contato", [x for x in ORDEM_MESES if x in base["mes_contato"].unique()], index=2)
        previo_s = c7.toggle("Lead já teve contato prévio", value=hist_s != "nonexistent")

        resolucao = 34
        gx = np.linspace(base[vx].quantile(0.01), base[vx].quantile(0.99), resolucao)
        gy = np.linspace(base[vy].quantile(0.01), base[vy].quantile(0.99), resolucao)
        if vx in ("idade", "n_contatos_campanha", "n_contatos_anteriores"):
            gx = np.unique(np.round(gx))
        if vy in ("idade", "n_contatos_campanha", "n_contatos_anteriores"):
            gy = np.unique(np.round(gy))
        malha_x, malha_y = np.meshgrid(gx, gy)
        fixos = {"idade": 38, "n_contatos_campanha": 1, "n_contatos_anteriores": 1 if previo_s else 0,
                 "taxa_variacao_emprego": float(base["taxa_variacao_emprego"].median()),
                 "indice_precos_consumidor": float(base["indice_precos_consumidor"].median()),
                 "indice_confianca_consumidor": float(base["indice_confianca_consumidor"].median())}
        atributos = {v: np.full(malha_x.size, valor) for v, valor in fixos.items()}
        atributos[vx], atributos[vy] = malha_x.ravel(), malha_y.ravel()
        atributos.update({"profissao": prof_s, "estado_civil": "married", "inadimplencia": "no",
                          "canal_contato": canal_s, "mes_contato": mes_s, "resultado_campanha_anterior": hist_s,
                          "escolaridade": "university.degree", "contato_previo": int(previo_s),
                          "dias_desde_ultimo_contato": float(modelo["mediana_pdays"])})
        _, prob_malha = pontuar(montar_leads(atributos))
        superficie = prob_malha.reshape(malha_x.shape) * 100
        fig = go.Figure(go.Surface(
            x=gx, y=gy, z=superficie, colorscale=ESCALA, colorbar=dict(title="%", len=0.6),
            contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="#FFFFFF", project_z=True)),
            hovertemplate=f"{ROTULOS.get(vx, vx)}: %{{x:.1f}}<br>{ROTULOS.get(vy, vy)}: %{{y:.1f}}"
                          "<br>Probabilidade: %{z:.1f}%<extra></extra>"))
        st.plotly_chart(cena_3d(fig, ROTULOS.get(vx, vx), ROTULOS.get(vy, vy), "Probabilidade (%)", 640),
                        width="stretch")
        pico = np.unravel_index(np.argmax(superficie), superficie.shape)
        s1, s2, s3 = st.columns(3)
        s1.metric("Probabilidade máxima na superfície", pct(superficie.max() / 100),
                  f"{ROTULOS.get(vx, vx)} = {gx[pico[1]]:.1f} · {ROTULOS.get(vy, vy)} = {gy[pico[0]]:.1f}"
                  .replace(".", ","), delta_color="off")
        s2.metric("Probabilidade mínima", pct(superficie.min() / 100))
        s3.metric("Amplitude", f"{superficie.max() - superficie.min():.1f} p.p.".replace(".", ",", 1),
                  "efeito conjunto das duas variáveis", delta_color="off")
        nota("<b>Como ler.</b> A superfície é calculada na hora pelo modelo treinado: as duas variáveis escolhidas "
             "variam e as demais ficam fixas no perfil selecionado acima (demais atributos: 38 anos, casado, "
             "ensino superior, sem inadimplência, valores macroeconômicos na mediana). Degraus são típicos de "
             "modelos baseados em árvores.")

    # ---- superfície do CAC
    with aba_cac:
        c1, c2, c3 = st.columns([1, 1, 1.4])
        cpl_s = c1.slider("Custo por lead - CPL (R$)", 0.0, 100.0, float(res["politica"]["cpl"]), step=1.0)
        custo_max = c2.slider("Custo por abordagem até (R$)", 5.0, 80.0, 40.0, step=5.0)
        modo_cac = c3.radio("Como a fila é formada", ["Fila única (teto do ganho)", "Fila do dia (piso do ganho)"],
                            horizontal=True)
        capacidades = np.arange(0.05, 1.0001, 0.05)
        custos = np.linspace(0, custo_max, 25)
        y_t, s_t = teste["converteu"].to_numpy(), teste["score_bruto"].to_numpy()
        if modo_cac.startswith("Fila única"):
            capturado = np.array([revocacao_no_topo(y_t, s_t, c) for c in capacidades])
        else:
            capturado = np.array([revocacao_intracontexto(teste, c) for c in capacidades])
        n_t, total_t = len(teste), y_t.sum()
        malha_c, malha_k = np.meshgrid(capacidades, custos)
        custo_total = n_t * cpl_s + np.ceil(malha_c * n_t) * malha_k
        cac_modelo = custo_total / (capturado[None, :] * total_t)
        cac_chegada = custo_total / (malha_c * total_t)
        fig = go.Figure()
        fig.add_surface(x=capacidades * 100, y=custos, z=cac_chegada, name="Ordem de chegada", showscale=False,
                        colorscale=[[0, "#55607F"], [1, "#B7C0D8"]], opacity=0.55,
                        hovertemplate="Capacidade: %{x:.0f}%<br>Custo por abordagem: R$ %{y:.0f}"
                                      "<br>CAC por ordem de chegada: R$ %{z:.0f}<extra></extra>")
        fig.add_surface(x=capacidades * 100, y=custos, z=cac_modelo, name="Fila priorizada",
                        colorscale=[[0, VERDE], [0.5, CIANO], [1, VIOLETA]], colorbar=dict(title="R$", len=0.6),
                        hovertemplate="Capacidade: %{x:.0f}%<br>Custo por abordagem: R$ %{y:.0f}"
                                      "<br>CAC com fila priorizada: R$ %{z:.0f}<extra></extra>")
        st.plotly_chart(cena_3d(fig, "Capacidade (%)", "Custo por abordagem (R$)", "CAC (R$)", 640), width="stretch")
        reducao = (1 - cac_modelo / cac_chegada)[0]
        melhor = int(np.argmax(reducao))
        r1, r2, r3 = st.columns(3)
        r1.metric("Maior redução do CAC", pct(reducao[melhor]), f"com capacidade de {capacidades[melhor]:.0%}",
                  delta_color="off")
        r2.metric("Redução com 20% de capacidade", pct(reducao[3]))
        r3.metric("Redução com 50% de capacidade", pct(reducao[9]))
        nota("<b>Como ler.</b> A superfície colorida é o CAC com a fila priorizada; a cinza translúcida, o CAC por "
             "ordem de chegada. A distância vertical entre elas é a economia por cliente. CPL e custo por abordagem "
             "são premissas ilustrativas; a redução percentual depende só do modelo.")


# ============================================================================= 3. DESEMPENHO
elif pagina.startswith("3"):
    m, ic = res["teste"]["metricas"], res["teste"]["ic95"]
    hero("AVALIAÇÃO", "Desempenho do modelo",
         f"{res['modelo_final']['algoritmo']} avaliado em {inteiro(res['particoes']['teste']['registros'])} leads "
         "que não participaram do treino, da calibração nem de qualquer escolha.")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("AUC-PR", dec(m["AUC-PR"]), f"sem modelo: {dec(res['teste']['prevalencia'])}", delta_color="off")
    k2.metric("AUC-ROC", dec(m["AUC-ROC"]), "sem modelo: 0,500", delta_color="off")
    k3.metric("Conversões nos 20% do topo", pct(m["Revocação top 20%"]), "sem modelo: 20,0%", delta_color="off")
    k4.metric("Lift no 1º decil", dec(m["Lift 1º decil"], 2) + "x", "sem modelo: 1,00x", delta_color="off")
    k5.metric("Erro de calibração (ECE)", dec(m["ECE"], 4), f"antes: {dec(m['ECE sem calibração'], 4)}",
              delta_color="off")
    st.caption("Intervalos de confiança de 95% (bootstrap): " + " · ".join(
        f"{nome} {dec(v[0], 3)} a {dec(v[1], 3)}" for nome, v in ic.items()))

    curvas = res["curvas_teste"]
    aba1, aba2, aba3, aba4, aba5 = st.tabs(["Curvas", "Comparação de modelos", "Desbalanceamento e calibração",
                                            "Variáveis que explicam o score", "Ablação e robustez"])
    with aba1:
        c1, c2, c3 = st.columns(3)
        fig = go.Figure()
        fig.add_scatter(x=curvas["pr"]["revocacao"], y=curvas["pr"]["precisao"], mode="lines", fill="tozeroy",
                        fillcolor="rgba(34,211,238,.10)", line=dict(color=CIANO, width=2.5), name="Modelo")
        fig.add_hline(y=res["teste"]["prevalencia"], line_dash="dot", line_color=CINZA)
        fig.update_xaxes(title="Revocação")
        fig.update_yaxes(title="Precisão", range=[0, 1])
        c1.plotly_chart(layout(fig, 370, "Precisão-revocação"), width="stretch")
        fig = go.Figure()
        fig.add_scatter(x=curvas["roc"]["fpr"], y=curvas["roc"]["tpr"], mode="lines", fill="tozeroy",
                        fillcolor="rgba(139,92,246,.10)", line=dict(color=VIOLETA, width=2.5), name="Modelo")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=CINZA, dash="dot"), name="Sem modelo")
        fig.update_xaxes(title="Taxa de falsos positivos")
        fig.update_yaxes(title="Taxa de verdadeiros positivos")
        c2.plotly_chart(layout(fig, 370, "ROC"), width="stretch")
        conf = curvas["confiabilidade"]
        fig = go.Figure()
        fig.add_scatter(x=conf["previsto_bruto"], y=conf["observado_bruto"], mode="lines+markers",
                        line=dict(color=AMBAR, dash="dash"), name="Sem calibração")
        fig.add_scatter(x=conf["previsto_calibrado"], y=conf["observado_calibrado"], mode="lines+markers",
                        line=dict(color=VERDE), name="Calibrado")
        fig.add_scatter(x=[0, 0.8], y=[0, 0.8], mode="lines", line=dict(color=CINZA, dash="dot"), name="Perfeita")
        fig.update_xaxes(title="Probabilidade prevista")
        fig.update_yaxes(title="Conversão observada")
        c3.plotly_chart(layout(fig, 370, "Confiabilidade (por decil)"), width="stretch")
    with aba2:
        st.write("Validação: partição usada para escolher o modelo. Teste: avaliação final, apenas para registro.")
        comparacao = pd.DataFrame(res["comparacao_modelos_validacao"]).T
        metrica = st.radio("Métrica no gráfico", ["AUC-PR", "AUC-ROC", "Revocação top 20%", "Lift 1º decil"],
                           horizontal=True)
        fig = go.Figure(go.Bar(x=comparacao.index, y=comparacao[metrica], marker_color=[CINZA, VIOLETA, CIANO],
                               text=[dec(v, 3) for v in comparacao[metrica]], textposition="outside"))
        fig.update_yaxes(range=[comparacao[metrica].min() * 0.9, comparacao[metrica].max() * 1.05])
        st.plotly_chart(layout(fig, 330, f"{metrica} na validação"), width="stretch")
        st.markdown("**Na validação**")
        st.dataframe(comparacao.style.format("{:.4f}"), width="stretch")
        st.markdown("**No teste**")
        st.dataframe(pd.DataFrame(res["teste"]["candidatos"]).T.style.format("{:.4f}"), width="stretch")
        with st.expander("Hiperparâmetros selecionados por validação cruzada"):
            st.json(res["hiperparametros"])
    with aba3:
        estrategias = pd.DataFrame(res["comparacao_estrategias_validacao"]).rename(
            columns={"algoritmo": "Algoritmo", "estrategia": "Estratégia"})
        xg = estrategias[estrategias["Algoritmo"] == "XGBoost"]
        fig = go.Figure()
        fig.add_bar(x=xg["Estratégia"], y=xg["Prob. média prevista"] * 100, name="Probabilidade média prevista",
                    marker_color=[VERDE, AMBAR, VERMELHO], text=[pct(v) for v in xg["Prob. média prevista"]],
                    textposition="outside")
        fig.add_hline(y=TAXA_GERAL * 100, line_dash="dash", line_color="#FFFFFF",
                      annotation_text="taxa real de conversão")
        fig.update_yaxes(title="%", range=[0, 48])
        st.plotly_chart(layout(fig, 330, "Ponderação e SMOTE inflam a probabilidade (XGBoost, validação)"),
                        width="stretch")
        st.dataframe(estrategias.style.format({c: "{:.4f}" for c in estrategias.columns[2:]}),
                     width="stretch", hide_index=True)
        st.markdown("**Calibração (validação cruzada dentro da validação)**")
        st.dataframe(pd.DataFrame(res["calibracao_validacao"]).T.style.format("{:.5f}"), width="stretch")
    with aba4:
        importancia = pd.Series(res["shap_importancia"]).sort_values().tail(12)
        fig = go.Figure(go.Bar(y=[ROTULOS.get(k, k) for k in importancia.index], x=importancia.values,
                               orientation="h", marker=dict(color=importancia.values, colorscale=ESCALA)))
        fig.update_xaxes(title="Média do valor absoluto de Shapley (log-chances)")
        st.plotly_chart(layout(fig, 440, "Importância global das variáveis (conjunto de teste)"), width="stretch")
    rt = res["robustez_temporal"]
    with aba5:
        st.write("Quanto do desempenho vem do perfil do lead e quanto vem do contexto em que ele chegou? "
                 "A AUC-ROC intracontexto compara apenas leads do mesmo mês e do mesmo cenário macroeconômico, "
                 "que é a situação da fila de um dia.")
        ablacao = pd.DataFrame(res["ablacao"]).set_index("Conjunto de variáveis")
        fig = go.Figure()
        fig.add_bar(x=ablacao.index, y=ablacao["AUC-ROC"], name="AUC-ROC global", marker_color=CIANO)
        fig.add_bar(x=ablacao.index, y=ablacao["AUC-ROC intracontexto"], name="AUC-ROC intracontexto",
                    marker_color=VIOLETA)
        fig.add_hline(y=0.5, line_dash="dot", line_color=CINZA)
        fig.update_layout(barmode="group")
        fig.update_yaxes(range=[0.4, 0.9], title="AUC-ROC")
        st.plotly_chart(layout(fig, 380, "Ablação de variáveis (conjunto de teste)"), width="stretch")
        st.dataframe(ablacao.style.format({c: "{:.4f}" for c in ablacao.columns if c != "Nº de variáveis"}),
                     width="stretch")
        st.markdown("**Partição temporal (treino nos primeiros 80%, teste nos 20% finais)**")
        st.dataframe(pd.Series(rt).to_frame("valor").style.format("{:.4f}"), width="stretch")
    nota(f"<b>Limitação: partição temporal.</b> Treinando nos primeiros 80% dos registros e testando nos 20% finais, "
         f"a AUC-ROC cai para {dec(rt['AUC-ROC'])} e a AUC-PR fica em {dec(rt['AUC-PR'])} (prevalência de "
         f"{dec(rt['taxa_conversao_teste'])} no período final, contra {dec(rt['taxa_conversao_treino'])} no treino). "
         "Os números acima são um limite superior do desempenho em produção; o uso real exige monitoramento e "
         "reciclagem do modelo.", "alerta")


# ============================================================================= 4. SIMULADOR
elif pagina.startswith("4"):
    hero("SIMULADOR", "Política de priorização",
         "Defina quanto da fila a equipe consegue atender e os custos da operação. O simulador compara a fila "
         "ordenada pelo modelo com o atendimento por ordem de chegada, no conjunto de teste.")

    c1, c2, c3 = st.columns(3)
    capacidade = c1.slider("Capacidade de atendimento (% dos leads)", 5, 100, 20, step=5) / 100
    cpl = c2.number_input("Custo por lead - CPL (R$)", 0.0, 1000.0, float(res["politica"]["cpl"]), step=1.0)
    custo_abordagem = c3.number_input("Custo por abordagem (R$)", 0.0, 1000.0,
                                      float(res["politica"]["custo_abordagem"]), step=1.0)
    c4, c5 = st.columns(2)
    canal = c4.multiselect("Restringir a canais", sorted(teste["canal_contato"].unique()), placeholder="Todos")
    profissao = c5.multiselect("Restringir a profissões", sorted(teste["profissao"].unique()), placeholder="Todos")
    modo = st.radio("Como a fila é formada", ["Fila única (leads de todos os períodos)",
                                              "Fila do dia (mesma capacidade dentro de cada contexto)"],
                    horizontal=True,
                    help="Fila única: o modelo escolhe os melhores leads de toda a base, o que inclui escolher os "
                         "melhores períodos. É o teto do ganho. Fila do dia: a capacidade é aplicada dentro de cada "
                         "grupo de leads do mesmo mês e do mesmo cenário macroeconômico. É a estimativa conservadora.")
    intracontexto = modo.startswith("Fila do dia")
    st.caption("CPL e custo por abordagem são premissas ilustrativas: a base não traz custos. "
               "A redução percentual do CAC depende só do modelo, e não dessas premissas.")

    fila = teste
    if canal:
        fila = fila[fila["canal_contato"].isin(canal)]
    if profissao:
        fila = fila[fila["profissao"].isin(profissao)]
    if len(fila) < 50 or fila["converteu"].sum() == 0:
        st.warning("O recorte tem leads ou conversões insuficientes para a simulação.")
        st.stop()

    y, score = fila["converteu"].to_numpy(), fila["score_bruto"].to_numpy()
    n, total = len(fila), y.sum()
    atendidos = int(np.ceil(capacidade * n))

    def capturado(fracao):
        return revocacao_intracontexto(fila, fracao) if intracontexto else revocacao_no_topo(y, score, fracao)

    conv_modelo = capturado(capacidade) * total
    conv_chegada = capacidade * total
    custo_total = n * cpl + atendidos * custo_abordagem
    cac_modelo, cac_chegada = custo_total / conv_modelo, custo_total / conv_chegada
    corte = np.sort(fila["prob_conversao"].to_numpy())[::-1][atendidos - 1]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Leads atendidos", inteiro(atendidos), f"de {inteiro(n)}", delta_color="off")
    k2.metric("Conversões com a fila priorizada", inteiro(conv_modelo),
              f"{conv_modelo - conv_chegada:+.0f} vs. chegada ({inteiro(conv_chegada)})")
    k3.metric("CAC com a fila priorizada", brl(cac_modelo),
              f"{(cac_modelo / cac_chegada - 1) * 100:+.1f}% vs. {brl(cac_chegada)}".replace(".", ",", 1),
              delta_color="inverse")
    if intracontexto:
        k4.metric("Lift da priorização", dec(conv_modelo / conv_chegada, 2) + "x", "sobre a ordem de chegada",
                  delta_color="off")
    else:
        k4.metric("Probabilidade mínima atendida", pct(corte), "ponto de corte da fila", delta_color="off")

    esquerda, direita = st.columns(2)
    with esquerda:
        fracoes = np.linspace(0.01, 1, 100)
        ganho = [capturado(f) * 100 for f in fracoes]
        fig = go.Figure()
        fig.add_scatter(x=fracoes * 100, y=ganho, mode="lines", fill="tozeroy",
                        fillcolor="rgba(34,211,238,.10)", line=dict(color=CIANO, width=3), name="Fila priorizada")
        fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color=CINZA, dash="dot"),
                        name="Ordem de chegada")
        fig.add_scatter(x=[capacidade * 100], y=[conv_modelo / total * 100], mode="markers",
                        marker=dict(color=AMBAR, size=15, line=dict(color="#FFFFFF", width=2)),
                        name="Cenário escolhido")
        fig.update_xaxes(title="% dos leads atendidos")
        fig.update_yaxes(title="% das conversões capturadas")
        st.plotly_chart(layout(fig, 410, "Curva de ganho acumulado"), width="stretch")
    with direita:
        cenarios = []
        for c in np.arange(0.05, 1.0001, 0.05):
            cm = capturado(c) * total
            custo = n * cpl + int(np.ceil(c * n)) * custo_abordagem
            cenarios.append({"capacidade": c * 100, "Fila priorizada": custo / cm,
                             "Ordem de chegada": custo / (c * total)})
        cenarios = pd.DataFrame(cenarios)
        fig = go.Figure()
        fig.add_scatter(x=cenarios["capacidade"], y=cenarios["Ordem de chegada"], mode="lines",
                        line=dict(color="#B7C0D8", width=2.5), name="Ordem de chegada")
        fig.add_scatter(x=cenarios["capacidade"], y=cenarios["Fila priorizada"], mode="lines", fill="tonexty",
                        fillcolor="rgba(52,211,153,.12)", line=dict(color=VERDE, width=3), name="Fila priorizada")
        fig.add_vline(x=capacidade * 100, line_dash="dash", line_color=AMBAR)
        fig.update_xaxes(title="Capacidade de atendimento (% dos leads)")
        fig.update_yaxes(title="CAC (R$)")
        st.plotly_chart(layout(fig, 410, "CAC por cenário de capacidade (área = economia)"), width="stretch")

    st.subheader("Conversão por decil da fila")
    ordenada = fila.sort_values("score_bruto", ascending=False, kind="stable").reset_index(drop=True)
    ordenada["decil"] = np.minimum(np.arange(len(ordenada)) * 10 // len(ordenada) + 1, 10)
    decis = ordenada.groupby("decil").agg(Leads=("converteu", "size"), Conversões=("converteu", "sum"),
                                          taxa=("converteu", "mean"), prevista=("prob_conversao", "mean"))
    decis["Lift"] = decis["taxa"] / ordenada["converteu"].mean()
    decis["Ganho acumulado"] = decis["Conversões"].cumsum() / total
    d1, d2 = st.columns([1, 1.2])
    fig = go.Figure()
    fig.add_bar(x=decis.index, y=decis["taxa"] * 100, name="Conversão observada",
                marker=dict(color=decis["taxa"], colorscale=ESCALA))
    fig.add_scatter(x=decis.index, y=decis["prevista"] * 100, mode="lines+markers", name="Prevista pelo modelo",
                    line=dict(color="#FFFFFF", width=2))
    fig.update_xaxes(title="Decil (1 = topo da fila)", dtick=1)
    fig.update_yaxes(title="%")
    d1.plotly_chart(layout(fig, 340), width="stretch")
    decis = decis.rename(columns={"taxa": "Taxa de conversão", "prevista": "Probabilidade prevista"})
    d2.dataframe(decis.style.format({"Taxa de conversão": "{:.1%}", "Probabilidade prevista": "{:.1%}",
                                     "Lift": "{:.2f}", "Ganho acumulado": "{:.1%}"}), width="stretch", height=390)


# ============================================================================= 5. PONTUAR UM LEAD
elif pagina.startswith("5"):
    hero("SCORE EM TEMPO REAL", "Pontuar um lead",
         "Altere qualquer atributo e a probabilidade, a posição na fila e a explicação são recalculadas na hora.")

    def opcoes(coluna):
        return sorted(base[coluna].unique())

    entrada, saida = st.columns([1.05, 1])
    with entrada:
        c1, c2 = st.columns(2)
        idade = c1.slider("Idade", 17, 98, 38)
        profissao = c2.selectbox("Profissão", opcoes("profissao"), index=opcoes("profissao").index("admin."))
        estado_civil = c1.selectbox("Estado civil", opcoes("estado_civil"),
                                    index=opcoes("estado_civil").index("married"))
        escolaridade = c2.selectbox("Escolaridade", opcoes("escolaridade"),
                                    index=opcoes("escolaridade").index("university.degree"))
        inadimplencia = c1.selectbox("Inadimplência", opcoes("inadimplencia"),
                                     index=opcoes("inadimplencia").index("no"))
        canal = c2.selectbox("Canal de contato", opcoes("canal_contato"))
        mes = c1.selectbox("Mês do contato", [x for x in ORDEM_MESES if x in base["mes_contato"].unique()], index=2)
        historico = c2.selectbox("Resultado da campanha anterior", ["nonexistent", "failure", "success"])
        n_campanha = c1.slider("Contatos na campanha atual", 1, int(modelo["p99_campaign"]), 1)
        n_anteriores = c2.slider("Contatos em campanhas anteriores", 0, 7, 0)
        contato_previo = c1.toggle("Teve contato prévio", value=False)
        dias = c2.slider("Dias desde o último contato", 0, 27, int(modelo["mediana_pdays"]),
                         disabled=not contato_previo)
        with st.expander("Contexto macroeconômico (padrão: mediana da base)"):
            m1, m2, m3 = st.columns(3)
            emprego = m1.slider("Variação do emprego", -3.4, 1.4, float(base["taxa_variacao_emprego"].median()),
                                step=0.1)
            precos = m2.slider("Índice de preços", 92.2, 94.8, float(base["indice_precos_consumidor"].median()),
                               step=0.1)
            confianca = m3.slider("Confiança do consumidor", -50.8, -26.9,
                                  float(base["indice_confianca_consumidor"].median()), step=0.5)

    lead = montar_leads({
        "profissao": [profissao], "estado_civil": [estado_civil], "inadimplencia": [inadimplencia],
        "canal_contato": [canal], "mes_contato": [mes], "resultado_campanha_anterior": [historico],
        "escolaridade": [escolaridade], "idade": [idade], "n_contatos_campanha": [n_campanha],
        "dias_desde_ultimo_contato": [float(dias)], "n_contatos_anteriores": [n_anteriores],
        "taxa_variacao_emprego": [emprego], "indice_precos_consumidor": [precos],
        "indice_confianca_consumidor": [confianca], "contato_previo": [int(contato_previo)]})
    bruto, prob = pontuar(lead)
    percentil = float((teste["score_bruto"] < bruto[0]).mean())
    prioridade = "Alta" if percentil >= 0.8 else "Média" if percentil >= 0.5 else "Baixa"
    cor_prioridade = {"Alta": VERDE, "Média": AMBAR, "Baixa": VERMELHO}[prioridade]

    with saida:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob[0] * 100,
            number=dict(suffix="%", valueformat=".1f", font=dict(size=58, color="#FFFFFF")),
            gauge=dict(axis=dict(range=[0, 100], tickcolor=SUAVE), bar=dict(color=cor_prioridade, thickness=0.28),
                       bgcolor="rgba(0,0,0,0)", borderwidth=0,
                       steps=[dict(range=[0, modelo["limiar_f1"] * 100], color="#17203A"),
                              dict(range=[modelo["limiar_f1"] * 100, 100], color="#1D2A4D")],
                       threshold=dict(line=dict(color="#FFFFFF", width=3), thickness=0.8,
                                      value=TAXA_GERAL * 100))))
        fig = layout(fig, 430, "Probabilidade de conversão (linha branca = média da base)")
        fig.update_layout(separators=",.", margin=dict(l=45, r=45, t=60, b=30))
        st.plotly_chart(fig, width="stretch")
        nota(f"Limiar de abordagem: {pct(modelo['limiar_f1'])}. Acima dele, a faixa do medidor fica mais clara.")
    if True:
        k1, k2, k3 = st.columns(3)
        k1.metric("Em relação à média", f"{prob[0] / TAXA_GERAL:.1f}x".replace(".", ","),
                  f"média de {pct(TAXA_GERAL)}", delta_color="off")
        k2.metric("Posição na fila", f"top {max(1, round((1 - percentil) * 100))}%",
                  f"acima de {pct(percentil, 0)} dos leads", delta_color="off")
        k3.metric("Prioridade", prioridade,
                  "acima do limiar" if prob[0] >= modelo["limiar_f1"] else "abaixo do limiar", delta_color="off")

    contrib, valor_base = contribuicoes_shap(lead)
    linha = contrib.iloc[0]
    linha = linha[linha.abs() > 1e-9].sort_values(key=np.abs).tail(10)
    valores = {c: lead.iloc[0][c] for c in linha.index}
    fig = go.Figure(go.Bar(
        y=[f"{ROTULOS.get(c, c)} = {valores[c]}" for c in linha.index], x=linha.values, orientation="h",
        marker_color=[VERDE if v > 0 else VERMELHO for v in linha.values],
        hovertemplate="%{y}<br>Contribuição: %{x:+.3f} log-chances<extra></extra>"))
    fig.add_vline(x=0, line_color="#FFFFFF", line_width=1)
    fig.update_xaxes(title="Contribuição para as log-chances de conversão (valores de Shapley)")
    st.plotly_chart(layout(fig, 420, "Por que este lead recebeu esta pontuação"), width="stretch")
    st.caption("Barras verdes aumentam a probabilidade e vermelhas reduzem. A soma das contribuições com o valor de "
               f"base ({valor_base[0]:+.2f}) resulta nas log-chances do score antes da calibração. Atende ao direito "
               "de revisão de decisões automatizadas (LGPD, art. 20).")


# ============================================================================= 6. FILA PRIORIZADA
elif pagina.startswith("6"):
    hero("OPERAÇÃO", "Fila priorizada",
         "Leads do conjunto de teste ordenados pelo score. É a visão que a equipe comercial usaria no dia a dia.")

    c1, c2, c3, c4 = st.columns(4)
    quantidade = c1.slider("Tamanho da fila", 10, 500, 50, step=10)
    prob_minima = c2.slider("Probabilidade mínima (%)", 0, 90, 0, step=5) / 100
    canal = c3.multiselect("Canal de contato", sorted(teste["canal_contato"].unique()), placeholder="Todos")
    profissao = c4.multiselect("Profissão", sorted(teste["profissao"].unique()), placeholder="Todos")

    fila = teste[teste["prob_conversao"] >= prob_minima]
    if canal:
        fila = fila[fila["canal_contato"].isin(canal)]
    if profissao:
        fila = fila[fila["profissao"].isin(profissao)]
    fila = fila.sort_values("score_bruto", ascending=False, kind="stable").head(quantidade)
    if fila.empty:
        st.warning("Nenhum lead atende aos filtros selecionados.")
        st.stop()

    k1, k2, k3 = st.columns(3)
    k1.metric("Leads na fila", inteiro(len(fila)))
    k2.metric("Conversão observada na fila", pct(fila["converteu"].mean()),
              f"{fila['converteu'].mean() / teste['converteu'].mean():.1f}x a média do teste".replace(".", ",", 1),
              delta_color="off")
    k3.metric("Conversões previstas pelo modelo", dec(fila["prob_conversao"].sum(), 1),
              f"observadas: {int(fila['converteu'].sum())}", delta_color="off")

    colunas = ["id_lead", "prob_conversao", "converteu", "profissao", "idade", "canal_contato", "mes_contato",
               "resultado_campanha_anterior", "contato_previo", "n_contatos_campanha", "escolaridade"]
    exibir = fila[colunas].rename(columns={"id_lead": "Lead", "prob_conversao": "Probabilidade",
                                           "converteu": "Converteu (real)", **ROTULOS})
    exibir.insert(0, "Posição", range(1, len(exibir) + 1))
    st.dataframe(exibir, width="stretch", hide_index=True, height=520, column_config={
        "Probabilidade": st.column_config.ProgressColumn("Probabilidade", format="percent", min_value=0, max_value=1)})
    st.download_button("Baixar fila em CSV", exibir.to_csv(index=False).encode("utf-8-sig"),
                       "fila_priorizada.csv", "text/csv")


# ============================================================================= 7. ASSISTENTE DE IA
else:
    hero("ASSISTENTE DE IA", "Pergunte aos dados",
         "Converse sobre o modelo, os segmentos, a política de atendimento e as limitações do trabalho. "
         "As respostas usam os mesmos números do notebook e do artigo.")

    def contexto_do_projeto():
        """Resumo numérico enviado ao assistente. Estável entre perguntas, para aproveitar o cache de prompt."""
        def segmento(coluna):
            t = base.groupby(coluna)["converteu"].agg(["size", "mean"]).round(4)
            return {str(k): {"leads": int(v["size"]), "taxa_conversao": float(v["mean"])} for k, v in t.iterrows()}

        resumo = {
            "base": res["base"], "tratamento": res["tratamento"], "particoes": res["particoes"],
            "modelo_final": res["modelo_final"], "hiperparametros": res["hiperparametros"],
            "comparacao_modelos_validacao": res["comparacao_modelos_validacao"],
            "comparacao_estrategias_desbalanceamento_validacao": res["comparacao_estrategias_validacao"],
            "calibracao_validacao": res["calibracao_validacao"],
            "teste": {k: res["teste"][k] for k in ("metricas", "ic95", "prevalencia", "candidatos", "limiar")},
            "decis_teste": res["decis_teste"], "importancia_shap": res["shap_importancia"],
            "razao_de_chances_regressao_logistica": res["razao_chances"],
            "politica_fila_unica": res["politica"], "politica_teto_e_piso": res["politica_conservadora"],
            "ablacao": res["ablacao"], "robustez_temporal": res["robustez_temporal"],
            "conversao_por_segmento_base_completa": {c: segmento(c) for c in (
                "profissao", "canal_contato", "mes_contato", "resultado_campanha_anterior", "faixa_idade",
                "escolaridade", "estado_civil", "inadimplencia", "contato_previo")},
        }
        return json.dumps(resumo, ensure_ascii=False, sort_keys=True)

    INSTRUCOES = (
        "Você é o assistente analítico do dashboard 'Lead Scoring em Mídia Paga', trabalho final (P2) do Projeto "
        "Integrador IV do curso de Ciência de Dados e Inteligência Artificial do IESB, de Asafe Ferreira Gomes. "
        "Quem conversa com você é a banca avaliadora, o autor ou um gestor comercial.\n\n"
        "O projeto treina um XGBoost calibrado para estimar a probabilidade de conversão de um lead com atributos "
        "conhecidos na sua entrada, usando a base pública Bank Marketing (UCI) como análogo de uma operação de mídia "
        "paga. O score ordena a fila de atendimento para reduzir o custo de aquisição de clientes (CAC).\n\n"
        "Responda em português do Brasil, de forma direta, com os números do JSON abaixo, usando vírgula decimal. "
        "Esses números são a única fonte de fatos sobre o projeto: quando a pergunta pedir um dado que não está "
        "neles, diga que o dashboard não tem esse dado em vez de estimar. Explique conceitos (AUC-PR, calibração, "
        "SHAP, SMOTE, desvio de conceito) quando ajudar, em linguagem acessível. Seja franco sobre as limitações: "
        "o ganho em fila única é um teto, o ganho intracontexto é o piso, e o desempenho cai em partição temporal. "
        "Os custos (CPL e custo por abordagem) são premissas ilustrativas. Respostas curtas, sem títulos; use uma "
        "lista só quando houver vários itens paralelos.\n\nDados do projeto (JSON):\n")

    def responder_local(pergunta):
        """Respostas por regras, usadas quando não há chave da API."""
        q = pergunta.lower()
        m, rt, abl = res["teste"]["metricas"], res["robustez_temporal"], res["ablacao"]
        teto = {round(c["capacidade"], 2): c for c in res["politica_conservadora"]}
        if any(t in q for t in ("cac", "custo", "capacidade", "economia", "priorizar", "priorização")):
            numero = re.search(r"(\d{1,3})\s*%", q)
            c = min(teto, key=lambda k: abs(k - int(numero.group(1)) / 100)) if numero else 0.2
            linha = teto[c]
            return (f"Com capacidade de {pct(c, 0)}, a fila priorizada captura "
                    f"{pct(linha['conversoes_capturadas_fila_unica'])} das conversões em fila única e "
                    f"{pct(linha['conversoes_capturadas_intracontexto'])} dentro do mesmo contexto. A redução do CAC "
                    f"fica entre {pct(linha['reducao_cac_intracontexto_pct'] / 100)} (piso, fila do dia) e "
                    f"{pct(linha['reducao_cac_fila_unica_pct'] / 100)} (teto, fila única). Os valores em reais "
                    "dependem das premissas de custo; a redução percentual depende só do modelo.")
        if any(t in q for t in ("limita", "temporal", "robust", "fraqueza", "problema", "risco", "contexto")):
            return (f"A principal limitação é a dependência do contexto. Em partição temporal a AUC-ROC cai de "
                    f"{dec(m['AUC-ROC'])} para {dec(rt['AUC-ROC'])}, porque a taxa de conversão passa de "
                    f"{pct(rt['taxa_conversao_treino'])} no treino para {pct(rt['taxa_conversao_teste'])} no período "
                    f"final. Entre leads do mesmo contexto, a AUC-ROC é de {dec(abl[0]['AUC-ROC intracontexto'])}. "
                    "Além disso, a base é de telemarketing bancário de 2008 a 2010 e os custos são premissas.")
        if any(t in q for t in ("variáve", "variave", "shap", "importan", "explica", "pesa")):
            topo = list(res["shap_importancia"].items())[:5]
            total = sum(res["shap_importancia"].values())
            return ("As variáveis de maior peso no score, pela média do valor absoluto de Shapley, são: "
                    + "; ".join(f"{ROTULOS.get(k, k)} ({pct(v / total, 0)})" for k, v in topo)
                    + ". O contexto macroeconômico domina, e o perfil declarado pesa menos.")
        if any(t in q for t in ("smote", "desbalance", "ponderação", "ponderacao", "limiar")):
            e = {x["estrategia"]: x for x in res["comparacao_estrategias_validacao"] if x["algoritmo"] == "XGBoost"}
            return (f"As três estratégias ordenam praticamente igual (AUC-PR de {dec(e['Sem tratamento']['AUC-PR'])}, "
                    f"{dec(e['Ponderação de classes']['AUC-PR'])} e {dec(e['SMOTE']['AUC-PR'])}). A diferença está na "
                    f"probabilidade: a taxa real é de {pct(TAXA_GERAL)}, e a ponderação prevê em média "
                    f"{pct(e['Ponderação de classes']['Prob. média prevista'])}; o SMOTE, "
                    f"{pct(e['SMOTE']['Prob. média prevista'])}. Por isso o modelo final foi treinado sem "
                    "reamostragem, com ajuste do limiar.")
        if any(t in q for t in ("calibra", "ece", "brier")):
            return (f"O modelo já saía próximo da calibração. Com a escala de Platt, o ECE no teste passou de "
                    f"{dec(m['ECE sem calibração'], 4)} para {dec(m['ECE'], 4)} e o Brier ficou em {dec(m['Brier'], 4)}. "
                    "Na prática, a probabilidade prevista erra em média pouco mais de um ponto percentual.")
        for coluna, termos in (("profissao", ("profiss", "ocupa")), ("canal_contato", ("canal", "celular", "telefone")),
                               ("mes_contato", ("mês", "mes ", "meses", "sazonal")),
                               ("resultado_campanha_anterior", ("histórico", "historico", "campanha anterior")),
                               ("faixa_idade", ("idade", "faixa", "jovem", "idoso")),
                               ("escolaridade", ("escolaridade", "ensino"))):
            if any(t in q for t in termos):
                t = base.groupby(coluna)["converteu"].agg(["size", "mean"]).sort_values("mean", ascending=False)
                return (f"Conversão por {ROTULOS.get(coluna, coluna).lower()} (média geral de {pct(TAXA_GERAL)}): "
                        + "; ".join(f"{k}: {pct(v['mean'])} ({inteiro(v['size'])} leads)" for k, v in t.iterrows())
                        + ".")
        if "desempenho" not in q and any(t in q for t in ("algoritmo", "xgboost", "floresta", "logística",
                                                         "logistica", "compar", "qual modelo", "melhor modelo")):
            v = res["comparacao_modelos_validacao"]
            return ("Na validação, os três algoritmos ficaram próximos: "
                    + "; ".join(f"{nome} com AUC-PR de {dec(x['AUC-PR'])}" for nome, x in v.items())
                    + ". O XGBoost foi escolhido pela pequena vantagem e por ter valores de Shapley exatos.")
        return (f"O modelo final é um XGBoost calibrado. No teste: AUC-PR de {dec(m['AUC-PR'])} (sem modelo: "
                f"{dec(res['teste']['prevalencia'])}), AUC-ROC de {dec(m['AUC-ROC'])}, "
                f"{pct(m['Revocação top 20%'])} das conversões nos 20% do topo da fila e lift de "
                f"{dec(m['Lift 1º decil'], 2)} no primeiro decil. Em modo local eu respondo sobre desempenho, CAC e "
                "capacidade, variáveis, desbalanceamento, calibração, limitações e conversão por segmento.")

    def responder_claude(historico, chave):
        """Resposta em streaming pela API do Claude. Devolve um gerador de texto."""
        import anthropic

        cliente = anthropic.Anthropic(api_key=chave) if chave else anthropic.Anthropic()
        try:
            with cliente.beta.messages.stream(
                model=MODELO_IA, max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                output_config={"effort": "medium"},
                cache_control={"type": "ephemeral"},
                system=INSTRUCOES + contexto_do_projeto(),
                messages=historico,
            ) as fluxo:
                yield from fluxo.text_stream
                final = fluxo.get_final_message()
            if final.stop_reason == "refusal":
                yield "\n\nO modelo recusou esta solicitação. Reformule a pergunta."
            elif final.stop_reason == "max_tokens":
                yield "\n\n(Resposta interrompida pelo limite de tamanho.)"
        except anthropic.AuthenticationError:
            yield "A chave da API foi rejeitada. Confira a chave na barra lateral."
        except anthropic.RateLimitError:
            yield "Limite de requisições atingido. Aguarde alguns segundos e pergunte de novo."
        except anthropic.APIStatusError as erro:
            yield f"A API retornou o erro {erro.status_code}. Tente novamente."
        except anthropic.APIConnectionError:
            yield "Não foi possível conectar à API. Verifique a internet."

    try:
        chave_segredo = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        chave_segredo = ""
    chave_ambiente = os.environ.get("ANTHROPIC_API_KEY", "") or chave_segredo
    with st.sidebar.expander("Assistente de IA"):
        chave_digitada = st.text_input("Chave da API do Claude", type="password",
                                       help="Opcional. Fica só na memória desta sessão e não é gravada em disco.")
    chave = chave_digitada or chave_ambiente
    usa_claude = bool(chave) or bool(os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    if usa_claude:
        st.markdown(f'<span class="pilula ok">Claude conectado · {MODELO_IA}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pilula local">Modo local · sem chave da API</span>', unsafe_allow_html=True)
        st.markdown('<div class="cartao-ia" style="margin-top:8px">Sem chave, o assistente responde por regras '
                    'sobre os dados do projeto: desempenho, CAC e capacidade, variáveis, desbalanceamento, '
                    'calibração, limitações e conversão por segmento. Para respostas livres em linguagem natural, '
                    'informe uma chave da API do Claude na barra lateral.</div>', unsafe_allow_html=True)

    if "conversa" not in st.session_state:
        st.session_state.conversa = []

    sugestoes = ["Qual é o desempenho do modelo?", "Quanto o CAC cai com 30% de capacidade?",
                 "Quais variáveis mais pesam no score?", "Quais são as limitações do trabalho?",
                 "Por que não usar SMOTE?", "Qual profissão converte mais?"]
    pergunta_sugerida = None
    colunas_sugestao = st.columns(3)
    for k, texto in enumerate(sugestoes):
        if colunas_sugestao[k % 3].button(texto, key=f"sugestao_{k}", width="stretch"):
            pergunta_sugerida = texto

    for mensagem in st.session_state.conversa:
        with st.chat_message(mensagem["role"]):
            st.markdown(mensagem["content"])

    pergunta = st.chat_input("Pergunte sobre o modelo, os segmentos ou a política de atendimento") or pergunta_sugerida
    if pergunta:
        st.session_state.conversa.append({"role": "user", "content": pergunta})
        with st.chat_message("user"):
            st.markdown(pergunta)
        with st.chat_message("assistant"):
            if usa_claude:
                resposta = st.write_stream(responder_claude(list(st.session_state.conversa), chave))
            else:
                resposta = responder_local(pergunta)
                st.markdown(resposta)
        st.session_state.conversa.append({"role": "assistant", "content": resposta})

    if st.session_state.conversa and st.button("Limpar conversa"):
        st.session_state.conversa = []
        st.rerun()
