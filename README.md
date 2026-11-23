# Lead Scoring em Mídia Paga — Avaliação P2

Projeto Integrador Aplicado em Ciência de Dados e Inteligência Artificial IV — IESB
Aluno: Asafe Ferreira Gomes · Professora: Simone de Araújo Góes Assis

Continuação do projeto da P1. Estima a probabilidade de conversão de um lead a partir dos
atributos conhecidos na sua entrada, para ordenar a fila de atendimento comercial e reduzir
o custo de aquisição de clientes (CAC).

## Conteúdo da entrega

| Pasta | Produto |
|---|---|
| `01_Artigo/` | Artigo científico (Word, com equações editáveis, e PDF) |
| `02_Notebook/` | Notebook executável `Lead_Scoring_P2.ipynb`, já com as saídas |
| `03_Dashboard/` | Dashboard interativo em Streamlit (`app.py`), com explorador 3D e assistente de IA |
| `04_Apresentacao/` | Slides da apresentação oral |
| `dados/` | Base bruta (UCI) e base tratada |
| `artefatos/` | Modelo treinado, métricas e base pontuada gerados pelo notebook |
| `figuras/` | Figuras geradas pelo notebook e usadas no artigo e nos slides |
| `ferramentas/` | Scripts auxiliares que montam o notebook, o artigo e os slides |

## Como reproduzir

Requisitos: Python 3.10 ou superior (testado no Python 3.13, Windows 11).

```bash
# 1. Criar o ambiente e instalar as dependências
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 2. Executar o notebook (gera dados/tratado, artefatos e figuras)
jupyter notebook 02_Notebook/Lead_Scoring_P2.ipynb
#    Menu "Run" > "Run All Cells". Leva de 3 a 8 minutos.

# 3. Abrir o dashboard
streamlit run 03_Dashboard/app.py
#    O navegador abre em http://localhost:8501
```

A pasta `artefatos/` já vai preenchida, então o dashboard funciona sem reexecutar o notebook.

O assistente de IA do dashboard funciona sem configuração, em modo local (respostas por regras sobre os
dados). Para respostas livres em linguagem natural, defina a variável de ambiente `ANTHROPIC_API_KEY`
ou informe a chave da API do Claude na barra lateral; ela fica só na memória da sessão.
Se a base bruta não estiver em `dados/bruto/`, o notebook a baixa do UCI automaticamente.
No Google Colab, basta enviar o notebook e executar: as pastas são criadas na hora.

## Consistência entre os produtos

O notebook é a única fonte dos números. Ele grava `artefatos/resultados.json`, e o dashboard,
o artigo e os slides leem esse arquivo. Todas as etapas aleatórias usam a semente 42.

## Dados

Bank Marketing (bank-additional-full), UCI Machine Learning Repository, licença CC BY 4.0.
Moro, Cortez e Rita (2014). A base é anonimizada e não contém dados pessoais identificáveis.

## Uso de inteligência artificial

Ferramenta de IA generativa (Claude, da Anthropic) foi usada como apoio na programação, na
redação e na revisão. As decisões metodológicas, a verificação dos resultados e o conteúdo
final são de responsabilidade do autor.
