---
title: "Lead Scoring em Mídia Paga: Ciência de Dados Aplicada à Priorização Comercial e à Redução do Custo de Aquisição de Clientes"
author:
  - "Asafe Ferreira Gomes"
  - "Centro Universitário IESB — Curso de Ciência de Dados e Inteligência Artificial — Brasília, DF"
  - "Projeto Integrador Aplicado em Ciência de Dados e Inteligência Artificial IV — Avaliação P2"
  - "Orientação: Profa. Simone de Araújo Góes Assis"
---

# Resumo

Empresas que captam clientes por mídia paga geram leads em volume proporcional ao investimento, mas atendem com uma equipe comercial de capacidade fixa. Sem critério de priorização, o esforço é alocado por ordem de chegada e o custo de aquisição de clientes (CAC) sobe. Este trabalho desenvolve e avalia um modelo de *lead scoring* que estima a probabilidade de conversão de cada lead com atributos disponíveis no momento de sua entrada. Usou-se a base pública Bank Marketing, com @n_bruto@ contatos reais de marketing direto e taxa de conversão de @taxa_conv@, tratada segundo o diagnóstico da etapa anterior do projeto. Compararam-se regressão logística, floresta aleatória e *gradient boosting* (XGBoost), três estratégias para o desbalanceamento de classes e dois métodos de calibração, com particionamento estratificado em treino, validação e teste. O modelo final, um XGBoost calibrado, obteve no teste AUC-PR de @auc_pr@ (IC 95%: @auc_pr_lo@ a @auc_pr_hi@), contra @prev@ de um modelo sem informação, e AUC-ROC de @auc_roc@. Em fila única, os 20% de leads mais bem pontuados concentram @rec20@ das conversões, o que reduz o CAC em @red_cac_20@ em relação à ordem de chegada nesse cenário. A análise de ablação mostra, porém, que a maior parte desse ganho vem do contexto macroeconômico e temporal: entre leads contemporâneos, os 20% do topo capturam @cons20_rec@ das conversões, e em partição temporal a AUC-ROC cai para @rt_auc@. Conclui-se que a priorização por score é viável e produz ganho mensurável, mas que sua magnitude depende de como a fila é formada e exige monitoramento e reciclagem do modelo. O trabalho entrega também um notebook reproduzível e um dashboard interativo com simulador de política de atendimento.

**Palavras-chave:** lead scoring; aprendizado de máquina; classificação desbalanceada; calibração de probabilidades; custo de aquisição de clientes.

# 1. Introdução

## 1.1 Contextualização

Em operações que usam mídia paga digital como principal canal de aquisição, o anúncio raramente vende de forma direta. Ele captura um contato interessado, o lead, que depois é trabalhado por uma equipe comercial até a conversão. O resultado final depende, portanto, de duas etapas encadeadas: a eficiência da mídia e a eficiência do atendimento.

A primeira etapa já é fortemente instrumentada por dados. As plataformas de anúncios oferecem otimização automática de lances, públicos semelhantes e atribuição de conversões. A segunda etapa costuma permanecer artesanal: os leads entram em uma fila única e são atendidos por ordem de chegada, sem estimativa de seu potencial. Como o volume de leads cresce com a verba e a capacidade da equipe não acompanha esse ritmo, a razão entre leads recebidos e leads bem atendidos se deteriora à medida que o investimento aumenta.

## 1.2 Problema e questão de pesquisa

Na ausência de um critério objetivo de priorização, leads com alta propensão a comprar esperam atrás de contatos sem intenção real, e a equipe gasta tempo em abordagens de baixo retorno. O efeito agregado é um CAC maior do que aquele que a mesma verba e a mesma equipe produziriam com uma ordenação melhor. A questão de pesquisa, formulada na primeira etapa do projeto, é a seguinte:

> É possível estimar, com acurácia e calibração suficientes para uso operacional, a probabilidade de conversão de um lead a partir exclusivamente dos atributos disponíveis no momento de sua entrada, de modo a permitir a priorização do atendimento comercial e a consequente redução do custo de aquisição de clientes?

## 1.3 Objetivos

**Objetivo geral.** Desenvolver e validar uma solução de Ciência de Dados capaz de estimar a probabilidade de conversão de leads a partir dos atributos disponíveis na captação, de modo a permitir a priorização do atendimento comercial e a redução do CAC.

**Objetivos específicos.** (i) Consolidar a preparação da base e definir o conjunto preditivo livre de vazamento e de multicolinearidade severa; (ii) construir e comparar modelos de classificação supervisionada, de uma regressão logística de referência a algoritmos de *ensemble* baseados em árvores; (iii) comparar estratégias para o desbalanceamento de classes; (iv) avaliar os modelos com métricas adequadas a classes desbalanceadas e ao uso do score como ordenador de fila; (v) calibrar as probabilidades preditas; (vi) interpretar a contribuição de cada variável por valores de Shapley; (vii) traduzir o score em política de priorização e estimar o impacto sobre o CAC; (viii) verificar a robustez dos resultados; e (ix) disponibilizar a solução em notebook reproduzível e em dashboard interativo.

## 1.4 Justificativa e contribuições

O CAC é uma das métricas que definem a viabilidade de operações baseadas em mídia paga. Reduzir o custo de mídia depende de leilões externos e cada vez mais disputados. O ganho obtido pela priorização do atendimento é interno: não exige verba adicional, contratação nem troca de plataforma. Além disso, as soluções comerciais de pontuação de leads funcionam, em geral, como caixas-pretas acopladas a um CRM específico ou usam pesos arbitrados manualmente.

As contribuições deste trabalho são quatro. A primeira é um procedimento completo, auditável e reproduzível, do dado bruto à política de atendimento. A segunda é a evidência empírica de que, para um score que ordena fila, treinar sem reamostragem e ajustar o limiar é preferível a ponderar classes ou aplicar SMOTE. A terceira é a separação, por ablação e por uma métrica intracontexto, entre o ganho que vem do perfil do lead e o que vem do contexto em que ele chega, distinção que muda a estimativa de impacto no CAC. A quarta é um dashboard que permite simular a política sob diferentes capacidades e custos e explicar cada pontuação individual.

# 2. Referencial teórico e trabalhos relacionados

## 2.1 Predição de conversão em marketing direto

A predição de resposta em marketing direto é um campo consolidado da literatura de sistemas de apoio à decisão. O trabalho de referência para a base aqui utilizada é o de Moro, Cortez e Rita (2014), que aplicaram a metodologia CRISP-DM (WIRTH; HIPP, 2000) a campanhas de telemarketing de um banco português e compararam regressão logística, árvores de decisão, redes neurais e máquinas de vetores de suporte. Os autores relatam o melhor resultado com rede neural, com AUC de 0,80 e alcance de cerca de 79% dos convertidos ao selecionar a metade mais bem classificada dos clientes, em avaliação com os contatos mais recentes. Advertem ainda que a duração da ligação, embora muito correlacionada com o desfecho, só é conhecida depois do contato e não deve ser usada em modelos preditivos realistas. O estudo anterior do mesmo grupo (MORO; LAUREANO; CORTEZ, 2011) estabeleceu o desenho metodológico. Uma limitação desses trabalhos para o presente propósito é que se concentram na discriminação e não tratam da calibração das probabilidades nem da tradução do score em custo de aquisição.

No contexto entre empresas, D'Haen e Van den Poel (2013) propõem um arcabouço iterativo de aquisição de clientes guiado por modelos preditivos. Monat (2011) examina a conversão de leads industriais e conclui que atributos comportamentais e de origem superam os puramente demográficos. Os dois trabalhos reforçam a utilidade da pontuação de leads, mas usam bases proprietárias, o que impede a verificação independente.

## 2.2 Aprendizado sob desbalanceamento de classes

Quando a classe de interesse é rara, um classificador pode atingir alta acurácia sem qualquer capacidade preditiva. He e Garcia (2009) revisam as abordagens para o problema, que se dividem em métodos no nível dos dados (reamostragem) e no nível do algoritmo (custos ou pesos por classe). Chawla et al. (2002) propuseram a sobreamostragem sintética da classe minoritária (SMOTE), que se tornou referência. A literatura posterior aponta uma limitação relevante dessas técnicas: ao alterar a proporção das classes no treino, elas distorcem as probabilidades estimadas, o que conflita com o uso do score como probabilidade.

Sobre a avaliação, Davis e Goadrich (2006) estabelecem a relação entre as curvas ROC e de precisão-revocação, e Saito e Rehmsmeier (2015) demonstram que a curva ROC produz impressão otimista sob forte desbalanceamento, enquanto a curva de precisão-revocação é mais informativa. Fawcett (2006) sistematiza a análise ROC e sua interpretação probabilística.

## 2.3 Algoritmos de ensemble baseados em árvores

Breiman (2001) introduziu as florestas aleatórias, que combinam árvores treinadas em amostras *bootstrap* com sorteio de atributos a cada divisão. Friedman (2001) formulou o *gradient boosting* como otimização funcional por descida de gradiente, e Chen e Guestrin (2016) apresentaram o XGBoost, com regularização explícita e aproximação de segunda ordem da função de perda. Esses métodos permanecem como referência de desempenho em dados tabulares heterogêneos (HASTIE; TIBSHIRANI; FRIEDMAN, 2009), que é o caso de bases de leads.

## 2.4 Calibração de probabilidades

Um score usado para ordenar fila e para estimar o número esperado de conversões precisa ser calibrado: entre os leads com probabilidade prevista de 30%, cerca de 30% devem converter. Platt (1999) propôs ajustar uma função sigmoide aos scores, e Zadrozny e Elkan (2002) propuseram a regressão isotônica, não paramétrica. Niculescu-Mizil e Caruana (2005) compararam os dois métodos em vários algoritmos e mostraram que modelos treinados por máxima verossimilhança tendem a sair já próximos da calibração, enquanto florestas e *boosting* com outras perdas se beneficiam mais do ajuste. A qualidade da calibração é medida pelo escore de Brier (1950) e pelo erro de calibração esperado (NAEINI; COOPER; HAUSKRECHT, 2015).

## 2.5 Interpretabilidade

Lundberg e Lee (2017) unificaram diversos métodos de explicação sob os valores de Shapley (1953), que atribuem a cada variável sua contribuição marginal média para uma predição. Lundberg et al. (2020) desenvolveram um algoritmo exato e eficiente para modelos de árvores. A explicabilidade tem aqui dupla função: permite que a equipe comercial entenda e confie no score e atende ao artigo 20 da Lei Geral de Proteção de Dados (BRASIL, 2018), que assegura ao titular a revisão de decisões tomadas unicamente com base em tratamento automatizado.

## 2.6 Vazamento de dados e desvio de conceito

Kaufman et al. (2012) formalizam o vazamento de dados como o uso, no treino, de informação que não estará disponível no momento da predição, e mostram que ele produz desempenho ilusório. Gama et al. (2014) revisam o desvio de conceito, isto é, a mudança da relação entre atributos e desfecho ao longo do tempo, e os mecanismos de adaptação. Os dois temas são centrais para este trabalho: o primeiro orientou a exclusão da duração do contato, e o segundo fundamenta a verificação de robustez temporal.

## 2.7 Soluções de mercado e lacuna

As plataformas de anúncios (Meta Advantage+, Google Smart Bidding) otimizam quem vê o anúncio e atuam antes da captação. As soluções de CRM (Salesforce Einstein, HubSpot, RD Station) pontuam o lead depois da captação, mas operam como caixas-pretas ligadas a um ecossistema ou com pesos definidos manualmente. Permanece espaço para uma abordagem orientada a dados, auditável e independente de plataforma, em que os critérios de pontuação sejam estimados da base histórica, as decisões de tratamento sejam explícitas e as predições individuais sejam explicáveis. Em relação à literatura acadêmica, a lacuna explorada é a passagem da discriminação para a decisão: calibração, política sob restrição de capacidade e separação entre sinal do lead e sinal do contexto.

# 3. Metodologia

## 3.1 Delineamento da pesquisa

Trata-se de pesquisa aplicada, de abordagem quantitativa e natureza preditiva, com dados secundários observacionais. O trabalho segue o ciclo CRISP-DM. As fases de entendimento do negócio, entendimento dos dados e preparação foram realizadas na primeira etapa do projeto e são aqui consolidadas; as fases de modelagem, avaliação e implantação compõem esta etapa. Por se tratar de dados observacionais, os resultados sustentam conclusões sobre capacidade preditiva e não sobre relações de causa e efeito.

## 3.2 Fonte, estrutura e variáveis dos dados

Usou-se o conjunto Bank Marketing, variante *bank-additional-full*, mantido pelo UCI Machine Learning Repository e publicado por Moro, Cortez e Rita (2014) sob licença Creative Commons BY 4.0. Os dados descrevem campanhas de marketing direto conduzidas entre maio de 2008 e novembro de 2010, com @n_bruto@ registros e 21 variáveis. A unidade de análise é um contato comercial com um lead, e a variável-alvo é binária: adesão ou não ao produto ofertado. A taxa de conversão é de @taxa_conv@, o que corresponde a @razao@ não convertidos para cada convertido.

A base é usada como análogo estrutural de uma operação de mídia paga, e não como sua representação literal. A estrutura analítica é a mesma: um contato é captado, um atributo de origem é registrado, uma equipe realiza a abordagem e um desfecho binário é observado. O que se pretende transferir é o método, e não os coeficientes estimados. As variáveis se organizam em quatro blocos (Tabela 1).

| Bloco | Variáveis | Equivalente em mídia paga |
|:---|:---|:---|
| Perfil declarado | idade, profissão, estado civil, escolaridade, inadimplência, financiamento imobiliário, empréstimo pessoal | Dados do formulário de captação |
| Campanha atual | canal de contato, mês, dia da semana, número de contatos, duração do contato | Canal de origem, data e esforço de abordagem |
| Histórico | dias desde o último contato, contatos anteriores, resultado da campanha anterior | Histórico do lead no CRM |
| Contexto macroeconômico | variação do emprego, índice de preços, confiança do consumidor, Euribor de 3 meses, número de empregados | Sazonalidade e conjuntura |

: Tabela 1 — Blocos de variáveis da base e correspondência com o problema de negócio

A base é integralmente anonimizada e não contém identificadores diretos ou indiretos. Na aplicação a dados proprietários de um anunciante passam a incidir as exigências da LGPD, em especial base legal, minimização, transparência e direito de revisão de decisões automatizadas.

## 3.3 Preparação e tratamento de anomalias

A análise exploratória da primeira etapa identificou anomalias que não aparecem em verificações automáticas, pois a base não possui valores nulos explícitos. O tratamento aplicado está resumido na Tabela 2 e é detalhado na Seção 4.

| Etapa | Anomalia | Critério | Decisão |
|:---|:---|:---|:---|
| 1 | 12 linhas idênticas | Igualdade em todas as colunas | Remoção, com preservação da primeira ocorrência |
| 2 | Ausência codificada como texto em 6 variáveis (@n_unknown@ registros, @pct_unknown@) | Mecanismo de ausência não completamente aleatório (LITTLE; RUBIN, 2019) | Categoria preservada como valor explícito e criação de indicadores de ausência |
| 3 | Valor-sentinela 999 em dias desde o último contato (@n_sentinela@ registros, @pct_sentinela@) | O código indica inexistência de contato prévio | Indicador binário de contato prévio e imputação pela mediana dos contatados (6 dias) |
| 4 | Vazamento na duração do contato | Variável só existe após o desfecho (KAUFMAN et al., 2012) | Mantida como descritiva e excluída do conjunto preditivo |
| 5 | Valores extremos no número de contatos (até 56) | Critério de Tukey (1977) e julgamento de domínio | Limitação ao percentil 99 (14 contatos; @n_wins@ registros); idade e contatos anteriores preservados |
| 6 | Multicolinearidade entre indicadores macroeconômicos | Fator de inflação da variância de até @vif_antes@ | Remoção de Euribor e número de empregados; VIF máximo de @vif_depois@ |
| 7 | Padronização | — | Nomes em português, tipos explícitos e variáveis derivadas |

: Tabela 2 — Etapas de tratamento da base

A base tratada tem @n_tratado@ registros e 28 colunas, com retenção de 99,97% dos registros e taxa de conversão inalterada. O desbalanceamento de classes não foi tratado nesta fase, pois é característica legítima do fenômeno; sua compensação pertence à modelagem e é aplicada apenas ao treino.

## 3.4 Conjunto preditivo, particionamento e pré-processamento

Entram no modelo apenas atributos disponíveis antes da abordagem. Foram excluídas a duração do contato (vazamento), a Euribor e o número de empregados (multicolinearidade) e as variáveis de financiamento imobiliário, empréstimo pessoal e dia da semana, cuja associação com o alvo é desprezível (Seção 4). Variáveis derivadas que são função determinística de outras já incluídas (número do mês, trimestre e o indicador de inadimplência desconhecida, idêntico a uma categoria da variável original após a codificação) também ficaram de fora, para não duplicar colunas. Restaram @n_pred@ variáveis preditoras, que geram @n_matriz@ colunas na matriz de projeto.

O particionamento foi estratificado pela variável-alvo em três subconjuntos: treino (60%, @n_treino@ registros), validação (20%, @n_val@) e teste (20%, @n_teste@). O treino ajusta os modelos e os hiperparâmetros por validação cruzada; a validação serve para comparar estratégias, calibrar e escolher o limiar; o teste é usado uma única vez, na avaliação final.

As transformações foram encapsuladas em um *pipeline* ajustado somente com o treino, o que evita vazamento entre partições: codificação *one-hot* para variáveis nominais, codificação inteira ordenada para escolaridade e faixa etária, e padronização das variáveis numéricas para média zero e desvio-padrão unitário.

## 3.5 Fundamentação matemática

**Formulação do problema.** Seja $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ o conjunto de dados, em que $\mathbf{x}_i \in \mathbb{R}^{d}$ é o vetor de atributos do lead $i$ conhecidos em sua entrada e $y_i \in \{0, 1\}$ indica a conversão. O objetivo é estimar a probabilidade condicional

$$p(\mathbf{x}) = P(y = 1 \mid \mathbf{x}), \qquad (1)$$

e usá-la para ordenar os leads em ordem decrescente de $\hat{p}(\mathbf{x}_i)$.

**Regressão logística.** O modelo de referência assume que as log-chances de conversão são lineares nos atributos:

$$\hat{p}_i = \sigma\left(\beta_0 + \mathbf{x}_i^{\top}\boldsymbol{\beta}\right), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}, \qquad (2)$$

em que $\beta_0$ é o intercepto e $\boldsymbol{\beta} \in \mathbb{R}^{d}$ é o vetor de coeficientes. Os parâmetros são estimados pela minimização da entropia cruzada ponderada com penalização quadrática:

$$\min_{\beta_0, \boldsymbol{\beta}} \; \sum_{i=1}^{n} w_{y_i}\left[-y_i \ln \hat{p}_i - (1 - y_i)\ln(1 - \hat{p}_i)\right] + \frac{1}{2C}\lVert\boldsymbol{\beta}\rVert_2^2, \qquad w_k = \frac{n}{2\,n_k}, \qquad (3)$$

em que $n_k$ é o número de observações da classe $k$, $w_k$ é o peso da classe (igual a 1 quando não há ponderação) e $C > 0$ é o inverso da força de regularização. A quantidade $e^{\beta_j}$ é a razão de chances associada ao aumento de uma unidade em $x_j$. O modelo pressupõe independência entre observações, linearidade nas log-chances e ausência de multicolinearidade severa, o que motivou o tratamento da Etapa 6.

**Floresta aleatória.** A floresta combina $B$ árvores de decisão $T_b$, cada uma treinada em uma amostra *bootstrap* do treino e com sorteio de um subconjunto de atributos a cada divisão:

$$\hat{p}(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^{B} T_b(\mathbf{x}). \qquad (4)$$

Cada divisão é escolhida para minimizar a impureza de Gini dos nós resultantes, $G = 1 - \sum_{k \in \{0,1\}} \pi_k^2$, em que $\pi_k$ é a proporção da classe $k$ no nó.

***Gradient boosting* (XGBoost).** O modelo é uma soma de $M$ árvores de regressão construídas em sequência, em que cada árvore corrige o erro das anteriores:

$$F_M(\mathbf{x}) = F_0 + \eta\sum_{m=1}^{M} f_m(\mathbf{x}), \qquad \hat{p}(\mathbf{x}) = \sigma\left(F_M(\mathbf{x})\right), \qquad (5)$$

em que $\eta \in (0, 1]$ é a taxa de aprendizado. A função objetivo regularizada é

$$\mathcal{L} = \sum_{i=1}^{n} \ell\left(y_i, F(\mathbf{x}_i)\right) + \sum_{m=1}^{M}\Omega(f_m), \qquad \Omega(f) = \gamma\,T + \frac{\lambda}{2}\sum_{j=1}^{T} w_j^2, \qquad (6)$$

em que $\ell$ é a perda logarítmica, $T$ é o número de folhas da árvore, $w_j$ é o valor da folha $j$ e $\gamma$ e $\lambda$ são hiperparâmetros de regularização. Na iteração $m$, a perda é aproximada por expansão de Taylor de segunda ordem, com gradientes $g_i = \partial \ell / \partial F$ e hessianas $h_i = \partial^2 \ell / \partial F^2$ avaliados em $F_{m-1}(\mathbf{x}_i)$. Para uma estrutura de árvore fixa, o valor ótimo da folha $j$ e o ganho de uma divisão em nós esquerdo (E) e direito (D) são

$$w_j^{*} = -\frac{G_j}{H_j + \lambda}, \qquad \text{Ganho} = \frac{1}{2}\left[\frac{G_E^2}{H_E + \lambda} + \frac{G_D^2}{H_D + \lambda} - \frac{(G_E + G_D)^2}{H_E + H_D + \lambda}\right] - \gamma, \qquad (7)$$

em que $G_j = \sum_{i \in I_j} g_i$ e $H_j = \sum_{i \in I_j} h_i$ somam sobre as observações $I_j$ da folha (CHEN; GUESTRIN, 2016).

**Estratégias para o desbalanceamento.** Foram comparadas três vias. A ponderação de classes usa os pesos $w_k$ da Equação 3 (no XGBoost, o fator equivalente multiplica os gradientes da classe positiva pela razão $n_0/n_1$). O SMOTE gera observações sintéticas da classe minoritária por interpolação entre uma observação $\mathbf{x}_i$ e um de seus $k$ vizinhos mais próximos da mesma classe, $\mathbf{x}_{i}^{(v)}$:

$$\mathbf{x}_{\text{novo}} = \mathbf{x}_i + \delta\left(\mathbf{x}_{i}^{(v)} - \mathbf{x}_i\right), \qquad \delta \sim U(0, 1). \qquad (8)$$

A terceira via não altera o treino: a regra de decisão $\hat{y} = \mathbb{1}[\hat{p} \geq \tau]$ usa o limiar $\tau^{*}$ que maximiza a medida $F_1$ na validação.

**Calibração.** Seja $s_i$ o score bruto do modelo. A escala de Platt ajusta uma sigmoide às log-chances do score, e a regressão isotônica ajusta uma função não decrescente $g$ por mínimos quadrados:

$$\hat{p}^{\text{cal}}_i = \sigma\left(a\,\ln\frac{s_i}{1 - s_i} + b\right), \qquad \min_{g \,\uparrow} \sum_{i}\left(y_i - g(s_i)\right)^2, \qquad (9)$$

com parâmetros $a$ e $b$ estimados por máxima verossimilhança na partição de validação. Os dois mapeamentos são monotônicos e, portanto, não alteram a ordenação dos leads.

**Métricas de avaliação.** Com $VP$, $FP$ e $FN$ denotando verdadeiros positivos, falsos positivos e falsos negativos para um limiar $\tau$, definem-se precisão, revocação e $F_1$:

$$P = \frac{VP}{VP + FP}, \qquad R = \frac{VP}{VP + FN}, \qquad F_1 = \frac{2\,P\,R}{P + R}. \qquad (10)$$

A área sob a curva de precisão-revocação é calculada como precisão média, e a área sob a curva ROC equivale à probabilidade de um convertido receber score maior que um não convertido:

$$\text{AUC-PR} = \sum_{k}\left(R_k - R_{k-1}\right)P_k, \qquad \text{AUC-ROC} = P\left(s^{+} > s^{-}\right), \qquad (11)$$

em que $P_k$ e $R_k$ são a precisão e a revocação no $k$-ésimo limiar. A referência de um modelo sem informação é a prevalência da classe positiva para a AUC-PR e 0,5 para a AUC-ROC. Para o uso operacional, define-se a revocação no topo da fila e o *lift*: sendo $S_c$ o conjunto dos $\lceil c\,n \rceil$ leads de maior score,

$$R(c) = \frac{\sum_{i \in S_c} y_i}{\sum_{i=1}^{n} y_i}, \qquad \text{Lift}(c) = \frac{R(c)}{c}. \qquad (12)$$

A qualidade da probabilidade é medida pelo escore de Brier e pelo erro de calibração esperado (ECE), este calculado em $K = 10$ faixas de mesma frequência:

$$\text{Brier} = \frac{1}{n}\sum_{i=1}^{n}\left(\hat{p}_i - y_i\right)^2, \qquad \text{ECE} = \sum_{k=1}^{K}\frac{n_k}{n}\left|\bar{y}_k - \bar{p}_k\right|, \qquad (13)$$

em que $n_k$, $\bar{y}_k$ e $\bar{p}_k$ são o tamanho, a taxa de conversão observada e a probabilidade média prevista da faixa $k$.

**AUC-ROC intracontexto.** Para medir a ordenação entre leads contemporâneos, o teste é dividido em grupos $g$ de leads com o mesmo mês e os mesmos indicadores macroeconômicos, e as áreas de cada grupo são combinadas com peso igual ao número de pares de convertido e não convertido:

$$\text{AUC}_{\text{intra}} = \frac{\sum_{g} m_g\,\text{AUC}_g}{\sum_{g} m_g}, \qquad m_g = n_g^{+}\,n_g^{-}. \qquad (14)$$

**Valores de Shapley.** A contribuição da variável $j$ para a predição de uma observação $\mathbf{x}$ é

$$\phi_j(\mathbf{x}) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!\,\left(|F| - |S| - 1\right)!}{|F|!}\left[f_{S \cup \{j\}}(\mathbf{x}) - f_S(\mathbf{x})\right], \qquad (15)$$

em que $F$ é o conjunto de variáveis e $f_S$ é a predição esperada quando apenas as variáveis em $S$ são conhecidas. As contribuições são aditivas na escala de log-chances, $F_M(\mathbf{x}) = \phi_0 + \sum_{j} \phi_j(\mathbf{x})$, propriedade verificada numericamente no notebook.

**Custo de aquisição.** Sejam $N$ o número de leads, $V$ o total de conversões potenciais, $c$ a fração de leads que a equipe consegue atender, $\text{CPL}$ o custo de mídia por lead e $\kappa$ o custo de uma abordagem. O custo de aquisição sob a fila priorizada e sob a ordem de chegada, admitida como equivalente a uma escolha aleatória, é

$$\text{CAC}_{\text{modelo}}(c) = \frac{N\,\text{CPL} + c\,N\,\kappa}{R(c)\,V}, \qquad \text{CAC}_{\text{chegada}}(c) = \frac{N\,\text{CPL} + c\,N\,\kappa}{c\,V}. \qquad (16)$$

Como o custo é o mesmo nos dois casos, a redução relativa do CAC depende apenas do *lift* e não das premissas de custo:

$$\Delta_{\text{CAC}}(c) = 1 - \frac{\text{CAC}_{\text{modelo}}(c)}{\text{CAC}_{\text{chegada}}(c)} = 1 - \frac{1}{\text{Lift}(c)}. \qquad (17)$$

**Medidas usadas no diagnóstico dos dados.** A associação entre variáveis categóricas e o alvo foi medida pelo V de Cramér com correção de viés (BERGSMA, 2013), e a multicolinearidade pelo fator de inflação da variância:

$$V = \sqrt{\frac{\chi^2 / n}{\min(r - 1,\, k - 1)}}, \qquad \text{VIF}_j = \frac{1}{1 - R_j^2}, \qquad (18)$$

em que $r$ e $k$ são as dimensões da tabela de contingência e $R_j^2$ é o coeficiente de determinação da regressão de $x_j$ sobre as demais variáveis numéricas. Para variáveis numéricas usou-se a correlação rank-biserial derivada do teste U de Mann-Whitney.

**Incerteza.** Os intervalos de confiança de 95% das métricas no teste foram obtidos por *bootstrap* percentílico com 1.000 reamostragens (EFRON; TIBSHIRANI, 1993).

# 4. Análise exploratória e preparação dos dados

## 4.1 Qualidade e estrutura

A base bruta não possui valores nulos explícitos e tem apenas 12 linhas duplicadas, o que sugere qualidade elevada. A inspeção detalhada mostrou que essa impressão é enganosa. A ausência de informação está codificada como a categoria textual *unknown* em seis variáveis e atinge @pct_unknown@ dos registros, com concentração de 20,9% na variável de inadimplência. Uma incidência tão alta e concentrada indica que a ausência não é fortuita: a recusa em informar a situação de crédito carrega informação sobre o lead. Por isso a categoria foi preservada como valor explícito, em vez de imputada pela moda, que atribuiria a situação mais frequente justamente a quem pode estar na situação oposta.

A variável de dias desde o último contato assume o valor 999 em @pct_sentinela@ dos registros. O código não representa um prazo, e sim a inexistência de contato prévio. Tratada como numérica, a variável teria média de 962,5 dias, contra 6,0 dias entre os leads efetivamente contatados. A sentinela esconde um dos atributos mais informativos da base: leads com contato prévio convertem a 63,8%, contra 9,3% dos demais. A variável foi decomposta em um indicador binário de contato prévio e em um valor numérico imputado pela mediana, escolhida por robustez à assimetria.

## 4.2 Distribuição do alvo e heterogeneidade entre segmentos

A Figura 1 mostra o desbalanceamento e a dispersão da conversão entre segmentos. A taxa varia de 6,9% a 31,4% conforme a profissão e de 8,8% a 65,1% conforme o resultado da campanha anterior. Essa heterogeneidade é a condição que torna a pontuação de leads viável: se todos convertessem a taxas semelhantes, qualquer ordenação equivaleria à ordem de chegada.

![Figura 1 — Distribuição da variável-alvo e taxa de conversão por profissão e por resultado da campanha anterior](figuras/fig01_eda_alvo_segmentos.png){width=100%}

## 4.3 Associação com o alvo e seleção de variáveis

Como a hipótese de normalidade foi rejeitada para todas as variáveis numéricas, adotou-se um protocolo não paramétrico. Com mais de 41 mil observações, diferenças triviais alcançam significância estatística, de modo que a seleção se baseou no tamanho de efeito (Tabela 3).

@tab_assoc@

Três leituras orientam a modelagem. A duração do contato tem a maior associação da base e é exatamente a variável que não pode ser usada. Os indicadores macroeconômicos e o mês aparecem logo em seguida, o que antecipa o peso do contexto no modelo, discutido na Seção 6.7. E as variáveis de financiamento imobiliário, empréstimo pessoal e dia da semana têm efeito desprezível mesmo com amostra de altíssimo poder estatístico, o que autoriza sua exclusão.

## 4.4 Multicolinearidade e valores extremos

Três indicadores macroeconômicos medem facetas do mesmo ciclo conjuntural e têm correlações de Spearman superiores a 0,92 entre si, com fator de inflação da variância de até @vif_antes@. Optou-se pela remoção seletiva, e não pela redução de dimensionalidade por componentes principais, para preservar o significado econômico das variáveis, necessário à explicabilidade. Mantida apenas a taxa de variação do emprego entre as três, o VIF máximo caiu para @vif_depois@.

Quanto aos valores extremos, o critério de Tukey serviu para informar o julgamento, e não para substituí-lo. As idades acima de 70 anos foram preservadas por serem plausíveis e corresponderem ao segmento de aposentados, de alta conversão. O número de contatos na campanha, que chega a 56, foi limitado ao percentil 99. Os contatos anteriores foram preservados porque os valores não nulos são o próprio sinal de interesse.

# 5. Desenvolvimento e implementação da solução

## 5.1 Arquitetura

A solução é composta por três camadas que compartilham os mesmos artefatos. O notebook lê a base bruta, aplica o tratamento, treina e avalia os modelos e grava quatro arquivos: o modelo final com seu calibrador, a base pontuada, os valores de Shapley do teste e um arquivo com todas as métricas. O dashboard lê esses arquivos para exibir indicadores, simular a política e pontuar novos leads. O texto deste artigo teve seus números interpolados do mesmo arquivo de métricas. Esse desenho garante que os três produtos apresentem resultados idênticos.

O fluxo de processamento segue esta ordem: (1) carga e tratamento; (2) particionamento estratificado; (3) *pipeline* de pré-processamento ajustado no treino; (4) busca de hiperparâmetros por validação cruzada; (5) comparação de estratégias de desbalanceamento na validação; (6) escolha do modelo e calibração; (7) avaliação única no teste; (8) interpretação, simulação de política e verificações de robustez; e (9) exportação dos artefatos.

## 5.2 Ferramentas

A implementação usa Python 3.13 com as bibliotecas pandas e NumPy (manipulação de dados), SciPy e statsmodels (testes e VIF), scikit-learn 1.9 (PEDREGOSA et al., 2011) para pré-processamento, modelos lineares, floresta aleatória, validação e métricas, XGBoost 3.4 (CHEN; GUESTRIN, 2016), imbalanced-learn (LEMAÎTRE; NOGUEIRA; ARIDAS, 2017) para o SMOTE, Matplotlib e Plotly para visualização, Streamlit para o dashboard e, de forma opcional, a API do Claude para o assistente em linguagem natural. Todas as operações aleatórias usam semente fixa, e as versões exatas das dependências estão registradas no arquivo de requisitos.

## 5.3 Ajuste de hiperparâmetros e estratégia de validação

Os hiperparâmetros foram escolhidos por busca aleatória com validação cruzada estratificada de 5 partes dentro do treino, com a AUC-PR como critério (Tabela 4). O SMOTE, quando usado, foi inserido no *pipeline* depois do pré-processamento, de modo que a reamostragem ocorre apenas nas partes de treino de cada dobra e nunca nos dados de avaliação.

@tab_hiper@

A validação segue três níveis de separação. A validação cruzada no treino escolhe os hiperparâmetros. A partição de validação compara algoritmos e estratégias, ajusta o calibrador e define o limiar. O método de calibração foi escolhido por validação cruzada de 5 partes dentro da própria validação, para não escolher e avaliar o calibrador com os mesmos dados. O teste só foi consultado depois de todas essas decisões.

# 6. Resultados e discussão

## 6.1 Comparação dos algoritmos

A Tabela 5 compara os três algoritmos na validação. Os três têm desempenho próximo, com vantagem pequena e consistente do XGBoost na AUC-PR (@xgb_val_aucpr@, contra @rf_val_aucpr@ da floresta aleatória e @rl_val_aucpr@ da regressão logística). A proximidade entre a AUC-PR da validação cruzada e a da validação indica ausência de sobreajuste relevante.

@tab_modelos@

O fato de um modelo linear ficar a cerca de um ponto percentual dos *ensembles* é um resultado por si: a maior parte do sinal da base é aproximadamente aditiva nas log-chances, e as interações capturadas pelas árvores acrescentam pouco. O Brier e o ECE muito piores da regressão logística não refletem ordenação inferior, e sim o efeito da ponderação de classes sobre a escala das probabilidades, examinado a seguir.

A regressão logística permite ainda uma leitura direta em razão de chances, com as demais variáveis mantidas constantes. Ter contato prévio multiplica as chances de conversão por @rc_previo@, e um resultado positivo na campanha anterior, por @rc_sucesso@. O aumento de um desvio-padrão na taxa de variação do emprego multiplica as chances por @rc_emprego@, e o contato por telefone fixo, por @rc_telefone@ em relação ao celular. Entre os meses, março (@rc_marco@) e maio (@rc_maio@) ocupam os extremos.

## 6.2 Estratégias para o desbalanceamento

A Tabela 6 mostra que as três estratégias quase não alteram a capacidade de ordenar. No XGBoost, a AUC-PR sem tratamento é de @xgb_val_aucpr@; com SMOTE cai levemente para @aucpr_smote@. O que muda é a escala das probabilidades. A taxa real de conversão é de 11,3%, e o modelo sem tratamento prevê em média @prob_sem@. Com ponderação de classes, a média prevista sobe para @prob_pond@, e com SMOTE, para @prob_smote@. O escore de Brier acompanha essa distorção: passa de @brier_sem@ para @brier_pond@ com a ponderação.

@tab_estrategias@

A medida $F_1$ no limiar convencional de 0,5 parece favorecer a ponderação e o SMOTE, mas essa vantagem é um artefato do limiar. Com o limiar ajustado na validação, o XGBoost sem tratamento alcança $F_1$ de @f1_sem_ot@, contra @f1_pond_ot@ com ponderação e @f1_smote_ot@ com SMOTE. O resultado é coerente com a literatura revisada na Seção 2.2: reamostrar ou ponderar desloca o ponto de operação sem melhorar a separação entre as classes. Para um score que ordena fila e precisa ser lido como probabilidade, a via mais adequada é treinar com a distribuição original e ajustar o limiar. Esse foi o critério de escolha do modelo final: XGBoost sem reamostragem.

## 6.3 Calibração

A Tabela 7 mostra que o modelo final já sai próximo da calibração, o que é esperado para um modelo treinado com perda logarítmica e sem alteração da proporção de classes (NICULESCU-MIZIL; CARUANA, 2005). Os dois métodos reduzem levemente o ECE e não melhoram o Brier. Adotou-se a escala de Platt, que teve o menor Brier entre os dois calibradores, por ter apenas dois parâmetros e menor risco de sobreajuste. A etapa foi mantida por ter custo desprezível e proteger o uso do score como probabilidade, mas o ganho é marginal e isso deve ser dito com clareza: no teste, o ECE passou de @ece_antes@ para @ece@.

@tab_calibracao@

## 6.4 Desempenho no conjunto de teste

A Tabela 8 traz a avaliação final. A AUC-PR de @auc_pr@ é @razao_aucpr@ vezes a de um modelo sem informação, e o intervalo de confiança exclui com folga esse valor de referência. A AUC-ROC de @auc_roc@ significa que, sorteados um convertido e um não convertido, o modelo atribui score maior ao convertido em cerca de 80% das vezes. O ECE de @ece@ indica que a probabilidade prevista se afasta, em média, pouco mais de um ponto percentual da frequência observada.

@tab_teste@

Para registro, a Tabela 9 mostra os três algoritmos no teste. A floresta aleatória e o XGBoost são estatisticamente indistinguíveis, pois a diferença entre eles é muito menor que a amplitude do intervalo de confiança. A escolha do XGBoost, feita na validação, justifica-se também pela disponibilidade de um algoritmo exato de valores de Shapley e pelo menor custo de inferência.

@tab_candidatos@

![Figura 2 — Curva de precisão-revocação, curva ROC e diagrama de confiabilidade no conjunto de teste](figuras/fig02_curvas_teste.png){width=100%}

O resultado é compatível com a literatura. Moro, Cortez e Rita (2014) relatam AUC de 0,80 e alcance de cerca de 79% dos convertidos na metade mais bem classificada. Aqui, a AUC-ROC é de @auc_roc@ e a metade superior da fila concentra @ganho50@ das conversões. A comparação exige cautela, porque aqueles autores usaram um conjunto mais amplo de atributos, não disponível na versão pública da base, e avaliaram com os contatos mais recentes, desenho mais exigente que o particionamento aleatório, como mostra a Seção 6.7.

## 6.5 Ganho acumulado e limiar de decisão

A Tabela 10 e a Figura 3 respondem à pergunta operacional. O primeiro decil converte @dec1_taxa@, contra 11,3% da média, um *lift* de @lift1@. Os dois primeiros decis concentram @rec20@ das conversões, e os quatro primeiros, @ganho40@. A coluna de probabilidade média prevista acompanha de perto a taxa observada em cada decil, o que confirma a calibração: no primeiro decil, o modelo previu @dec1_prev@ e a conversão foi de @dec1_taxa@.

@tab_decis@

![Figura 3 — Curva de ganho acumulado e taxa de conversão por decil do score no conjunto de teste](figuras/fig03_ganho_decis.png){width=100%}

A separação está concentrada no topo. A partir do terceiro decil as taxas ficam todas abaixo da média e variam pouco entre si, de modo que o score distingue bem um grupo pequeno de leads muito promissores e discrimina mal o restante. Para a operação, isso sugere uma política de dois níveis, com atendimento prioritário e imediato do topo, em vez de uma ordenação fina de toda a fila.

Quando se exige uma regra binária, o limiar que maximiza o $F_1$ na validação é de @limiar@. No teste, essa regra aborda @lim_abord@ dos leads, com precisão de @lim_prec@ e revocação de @lim_rev@ ($F_1$ de @lim_f1@): quase metade das abordagens resulta em conversão, e mais da metade das conversões é alcançada.

## 6.6 Interpretabilidade

A Figura 4 mostra a importância global das variáveis pela média do valor absoluto de Shapley. A taxa de variação do emprego domina, com @shap_emprego_pct@ da importância total, e os três indicadores macroeconômicos somam @shap_macro_pct@. Seguem o canal de contato, o mês e o histórico de relacionamento. O perfil declarado, com profissão, idade e escolaridade, tem peso bem menor.

![Figura 4 — Importância global das variáveis e distribuição das contribuições por valores de Shapley](figuras/fig04_shap_global.png){width=100%}

Esse resultado qualifica a expectativa formada na análise exploratória. O resultado da campanha anterior era o preditor categórico de maior associação bivariada, mas no modelo sua importância é menor, porque é informativo para poucos leads (apenas 3,7% tiveram contato prévio) e porque parte de seu sinal é compartilhada com o indicador de contato prévio. A importância global mede o efeito médio sobre todos os leads; para o lead que tem histórico positivo, a contribuição individual é grande, como mostra a explicação do lead mais bem pontuado do teste, em que contato prévio e sucesso anterior estão entre as três maiores contribuições.

A dominância do contexto macroeconômico é coerente com a base: a conversão variou muito ao longo do período, e os indicadores funcionam como marcadores desse período. A consequência para o uso do score é examinada na Seção 6.7.

## 6.7 Robustez: ablação e partição temporal

Leads que chegam no mesmo dia compartilham o mesmo contexto macroeconômico. As variáveis de contexto, portanto, não ajudam a ordenar a fila daquele dia; ajudam a distinguir períodos bons de períodos ruins. Duas verificações medem quanto do desempenho da Seção 6.4 pertence a cada caso.

A primeira é a ablação de variáveis (Tabela 11). Sem o bloco macroeconômico, a AUC-ROC cai de @auc_roc@ para @auc_sem_macro@; sem o bloco e sem o mês, para @auc_perfil@, com AUC-PR de @aucpr_perfil@ e @rec20_perfil@ das conversões nos 20% do topo. O perfil, o canal e o histórico sozinhos ainda ordenam muito melhor que o acaso, embora parte desse desempenho também reflita o período, já que o canal e o histórico mudaram ao longo das campanhas.

@tab_ablacao@

A coluna de AUC-ROC intracontexto é a mais reveladora. Calculada dentro dos @n_grupos@ grupos de leads com o mesmo mês e os mesmos indicadores, ela é de @auc_intra@ para o modelo completo e praticamente não muda entre os três conjuntos de variáveis. Ou seja, entre leads contemporâneos a capacidade de ordenação é modesta, e a diferença entre @auc_roc@ e @auc_intra@ corresponde ao que o modelo ganha por saber quando o lead chegou, e não quem ele é.

A segunda verificação é a partição temporal. Treinando o mesmo algoritmo nos primeiros 80% dos registros e avaliando nos 20% finais, a AUC-ROC cai para @rt_auc@ e o *lift* do primeiro decil, para @rt_lift@. A taxa de conversão do período final é de @rt_taxa@, contra @rt_taxa_treino@ no treino, e o modelo prevê em média @rt_prob@: a calibração se perde, com Brier de @rt_brier@. Trata-se de um caso claro de desvio de conceito (GAMA et al., 2014), em que tanto a prevalência quanto os valores das variáveis macroeconômicas saem da região observada no treino.

Esses resultados não invalidam o modelo, mas delimitam seu uso. O desempenho da Seção 6.4 é um limite superior, válido quando se prioriza uma base acumulada de leads de vários períodos, como em um reaquecimento de carteira. Para a fila do dia, a referência correta é a medida intracontexto.

## 6.8 Política de priorização e impacto no CAC

A Tabela 12 simula a política em fila única, com as premissas ilustrativas de R\$ @cpl@ por lead e R\$ @custo_abord@ por abordagem. Com capacidade para atender 20% dos leads, a ordem de chegada produz @conv20_chegada@ conversões e a fila priorizada, @conv20_modelo@. O CAC cai de R\$ @cac20_chegada@ para R\$ @cac20_modelo@, redução de @red_cac_20@. O ganho diminui à medida que a capacidade aumenta e desaparece quando a equipe atende todos os leads, o que confirma que o valor do score vem da restrição de capacidade.

@tab_politica@

![Figura 5 — Conversões e custo de aquisição por cenário de capacidade, em fila única](figuras/fig05_politica_cac.png){width=100%}

Os valores em reais dependem das premissas de custo, que a base não fornece. A redução percentual, pela Equação 17, depende apenas do *lift*. E o *lift*, como mostrou a Seção 6.7, depende de como a fila é formada. A Tabela 13 apresenta por isso os dois limites. No cenário conservador, em que a mesma fração é atendida dentro de cada contexto, os 20% do topo capturam @cons20_rec@ das conversões, um *lift* de @cons20_lift@, e a redução do CAC é de @cons20_red@. Com capacidade de 40%, a captura é de @cons40_rec@ e a redução, de @cons40_red@.

@tab_conservador@

A leitura adequada para a decisão de negócio é a de um intervalo: com capacidade de 20%, a priorização reduz o CAC entre @cons20_red@ e @red_cac_20@, conforme a fila seja do dia ou acumulada. Mesmo o piso é um ganho relevante, obtido sem verba adicional. A simulação admite duas simplificações que devem ser registradas: a ordem de chegada foi tratada como escolha aleatória, e supôs-se que priorizar um lead não altera sua probabilidade de conversão. A segunda hipótese é conservadora, pois o atendimento mais rápido de leads promissores tende a aumentar a conversão, mas esse efeito causal não pode ser medido com dados observacionais.

## 6.9 Dashboard interativo

O dashboard, desenvolvido em Streamlit, tem sete seções. A visão executiva permite filtrar a base por profissão, canal, mês, histórico e idade e comparar a conversão observada com a prevista em cada segmento. O explorador 3D (Figura 6) reúne três visualizações que podem ser giradas e ampliadas: um mapa dos leads construído pela projeção em componentes principais dos valores de Shapley, no qual leads próximos receberam o score pelos mesmos motivos; a superfície de resposta do modelo, calculada na hora para duas variáveis à escolha; e a superfície do CAC em função da capacidade e do custo por abordagem. A seção de desempenho reúne as métricas, as curvas, as comparações e a análise de ablação. O simulador de priorização (Figura 7) permite variar a capacidade de atendimento, o custo por lead e o custo por abordagem, escolher entre fila única e fila do dia e ver o efeito sobre conversões e CAC. A seção de pontuação recalcula, a cada alteração de atributo, a probabilidade calibrada, a posição na fila e a explicação por valores de Shapley. A fila priorizada lista os leads do teste em ordem de score, com filtros e exportação. Por fim, um assistente em linguagem natural responde a perguntas sobre o modelo e os resultados a partir do mesmo arquivo de métricas; ele usa um modelo de linguagem (Claude, da Anthropic) quando há chave de acesso configurada e, sem ela, responde por regras sobre os mesmos dados. O dashboard usa o próprio modelo treinado para pontuar novos leads e não se limita a reproduzir gráficos estáticos.

![Figura 6 — Explorador 3D do dashboard: mapa dos leads pela projeção dos valores de Shapley](figuras/telas/tela_3d.png){width=100%}

![Figura 7 — Simulador de priorização do dashboard interativo](figuras/telas/tela3.png){width=100%}

## 6.10 Limitações

As limitações do trabalho são as seguintes. (i) A base descreve telemarketing bancário em Portugal entre 2008 e 2010, e não mídia paga digital; transfere-se o método, e os coeficientes não devem ser aplicados a outro contexto. (ii) O particionamento aleatório mistura períodos, e o desempenho cai em partição temporal; o modelo exige monitoramento de desvio e reciclagem periódica. (iii) O período coberto inclui a crise financeira de 2008, o que acentua o peso das variáveis macroeconômicas e pode não se repetir em outros períodos. (iv) O número de contatos na campanha é conhecido antes de cada abordagem, mas não no instante exato da captação; em uma implantação, o score seria recalculado a cada tentativa. (v) Os custos usados na simulação são premissas. (vi) Os dados são observacionais, de modo que as associações não devem ser lidas como efeitos causais, e os valores de Shapley explicam o modelo, e não o mundo. (vii) Não foi avaliada a equidade do score entre grupos, verificação necessária antes do uso com dados pessoais reais, dado que idade e estado civil entram no modelo.

# 7. Conclusões

A questão de pesquisa tem resposta afirmativa, com uma ressalva importante quanto à magnitude. É possível estimar a probabilidade de conversão de um lead apenas com atributos disponíveis em sua entrada: o modelo final alcançou AUC-PR de @auc_pr@, contra @prev@ de referência, AUC-ROC de @auc_roc@ e erro de calibração de @ece@, de modo que o score pode ser lido como probabilidade. A priorização por esse score reduz o custo de aquisição sem verba adicional. A ressalva é que a maior parte da capacidade de ordenação vem do contexto temporal e macroeconômico. Com capacidade de 20%, a redução do CAC fica entre @cons20_red@, para leads contemporâneos, e @red_cac_20@, para uma base acumulada.

Os objetivos específicos foram alcançados. O conjunto preditivo foi definido sem vazamento e sem multicolinearidade severa. Três algoritmos foram comparados, e o XGBoost superou por pequena margem a floresta aleatória e a regressão logística. Das três estratégias para o desbalanceamento, a mais adequada foi treinar com a distribuição original e ajustar o limiar, pois ponderação e SMOTE distorcem as probabilidades sem melhorar a ordenação. A calibração trouxe ganho marginal, já que o modelo saía próximo dela. Os valores de Shapley tornaram o score explicável e revelaram a dominância do contexto. A política de priorização foi simulada e disponibilizada em dashboard interativo.

A principal contribuição metodológica é a separação entre o sinal do lead e o sinal do contexto, pela ablação e pela métrica intracontexto. Sem ela, o trabalho reportaria apenas o resultado da fila única e superestimaria o ganho da operação diária. Fica também a evidência de que a escolha do algoritmo importa menos do que a qualificação dos dados e o desenho da avaliação: a diferença entre os três modelos é de cerca de um ponto percentual, enquanto a diferença entre os desenhos de avaliação é de mais de dez.

Como continuidade, propõe-se: aplicar o procedimento a uma base proprietária de mídia paga, com atributos de comportamento digital (origem do clique, criativo, tempo até o preenchimento do formulário), que tendem a discriminar melhor entre leads contemporâneos; adotar validação temporal deslizante e monitoramento de desvio com reciclagem programada; avaliar a equidade do score; e medir o efeito causal da priorização por experimento controlado, com atendimento priorizado em parte da fila e por ordem de chegada no restante.

# Declaração de uso de inteligência artificial

Em conformidade com as orientações de integridade acadêmica da disciplina, declara-se que uma ferramenta de inteligência artificial generativa (Claude, da Anthropic) foi utilizada como apoio na programação, na organização do texto e na revisão. As decisões metodológicas, a conferência dos resultados e o conteúdo final são de responsabilidade do autor.

# Referências

BERGSMA, W. A bias-correction for Cramér's V and Tschuprow's T. **Journal of the Korean Statistical Society**, v. 42, n. 3, p. 323-328, 2013.

BRASIL. Lei n. 13.709, de 14 de agosto de 2018. Lei Geral de Proteção de Dados Pessoais (LGPD). Brasília: Diário Oficial da União, 2018.

BREIMAN, L. Random forests. **Machine Learning**, v. 45, n. 1, p. 5-32, 2001.

BRIER, G. W. Verification of forecasts expressed in terms of probability. **Monthly Weather Review**, v. 78, n. 1, p. 1-3, 1950.

CHAWLA, N. V.; BOWYER, K. W.; HALL, L. O.; KEGELMEYER, W. P. SMOTE: synthetic minority over-sampling technique. **Journal of Artificial Intelligence Research**, v. 16, p. 321-357, 2002.

CHEN, T.; GUESTRIN, C. XGBoost: a scalable tree boosting system. In: ACM SIGKDD INTERNATIONAL CONFERENCE ON KNOWLEDGE DISCOVERY AND DATA MINING, 22., 2016, San Francisco. **Proceedings** [...]. New York: ACM, 2016. p. 785-794.

DAVIS, J.; GOADRICH, M. The relationship between precision-recall and ROC curves. In: INTERNATIONAL CONFERENCE ON MACHINE LEARNING, 23., 2006, Pittsburgh. **Proceedings** [...]. New York: ACM, 2006. p. 233-240.

D'HAEN, J.; VAN DEN POEL, D. Model-supported business-to-business prospect prediction based on an iterative customer acquisition framework. **Industrial Marketing Management**, v. 42, n. 4, p. 544-551, 2013.

EFRON, B.; TIBSHIRANI, R. J. **An introduction to the bootstrap**. New York: Chapman & Hall, 1993.

FAWCETT, T. An introduction to ROC analysis. **Pattern Recognition Letters**, v. 27, n. 8, p. 861-874, 2006.

FRIEDMAN, J. H. Greedy function approximation: a gradient boosting machine. **The Annals of Statistics**, v. 29, n. 5, p. 1189-1232, 2001.

GAMA, J.; ŽLIOBAITĖ, I.; BIFET, A.; PECHENIZKIY, M.; BOUCHACHIA, A. A survey on concept drift adaptation. **ACM Computing Surveys**, v. 46, n. 4, art. 44, 2014.

HASTIE, T.; TIBSHIRANI, R.; FRIEDMAN, J. **The elements of statistical learning**: data mining, inference, and prediction. 2. ed. New York: Springer, 2009.

HE, H.; GARCIA, E. A. Learning from imbalanced data. **IEEE Transactions on Knowledge and Data Engineering**, v. 21, n. 9, p. 1263-1284, 2009.

KAUFMAN, S.; ROSSET, S.; PERLICH, C.; STITELMAN, O. Leakage in data mining: formulation, detection, and avoidance. **ACM Transactions on Knowledge Discovery from Data**, v. 6, n. 4, art. 15, 2012.

LEMAÎTRE, G.; NOGUEIRA, F.; ARIDAS, C. K. Imbalanced-learn: a Python toolbox to tackle the curse of imbalanced datasets in machine learning. **Journal of Machine Learning Research**, v. 18, n. 17, p. 1-5, 2017.

LITTLE, R. J. A.; RUBIN, D. B. **Statistical analysis with missing data**. 3. ed. Hoboken: Wiley, 2019.

LUNDBERG, S. M.; LEE, S.-I. A unified approach to interpreting model predictions. In: ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS, 30., 2017, Long Beach. **Proceedings** [...]. 2017. p. 4765-4774.

LUNDBERG, S. M. et al. From local explanations to global understanding with explainable AI for trees. **Nature Machine Intelligence**, v. 2, p. 56-67, 2020.

MONAT, J. P. Industrial sales lead conversion modeling. **Marketing Intelligence & Planning**, v. 29, n. 2, p. 178-194, 2011.

MORO, S.; CORTEZ, P.; RITA, P. A data-driven approach to predict the success of bank telemarketing. **Decision Support Systems**, v. 62, p. 22-31, 2014.

MORO, S.; LAUREANO, R.; CORTEZ, P. Using data mining for bank direct marketing: an application of the CRISP-DM methodology. In: EUROPEAN SIMULATION AND MODELLING CONFERENCE, 2011, Guimarães. **Proceedings** [...]. Guimarães: EUROSIS, 2011. p. 117-121.

NAEINI, M. P.; COOPER, G. F.; HAUSKRECHT, M. Obtaining well calibrated probabilities using Bayesian binning. In: AAAI CONFERENCE ON ARTIFICIAL INTELLIGENCE, 29., 2015, Austin. **Proceedings** [...]. 2015. p. 2901-2907.

NICULESCU-MIZIL, A.; CARUANA, R. Predicting good probabilities with supervised learning. In: INTERNATIONAL CONFERENCE ON MACHINE LEARNING, 22., 2005, Bonn. **Proceedings** [...]. New York: ACM, 2005. p. 625-632.

PEDREGOSA, F. et al. Scikit-learn: machine learning in Python. **Journal of Machine Learning Research**, v. 12, p. 2825-2830, 2011.

PLATT, J. C. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In: SMOLA, A. J. et al. (ed.). **Advances in large margin classifiers**. Cambridge: MIT Press, 1999. p. 61-74.

SAITO, T.; REHMSMEIER, M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. **PLoS ONE**, v. 10, n. 3, e0118432, 2015.

SHAPLEY, L. S. A value for n-person games. In: KUHN, H. W.; TUCKER, A. W. (ed.). **Contributions to the theory of games**. Princeton: Princeton University Press, 1953. v. 2, p. 307-317.

TUKEY, J. W. **Exploratory data analysis**. Reading: Addison-Wesley, 1977.

UCI MACHINE LEARNING REPOSITORY. **Bank Marketing Data Set**. Irvine: University of California, School of Information and Computer Science. Disponível em: https://archive.ics.uci.edu/dataset/222/bank+marketing. Acesso em: 23 nov. 2026.

WIRTH, R.; HIPP, J. CRISP-DM: towards a standard process model for data mining. In: INTERNATIONAL CONFERENCE ON THE PRACTICAL APPLICATIONS OF KNOWLEDGE DISCOVERY AND DATA MINING, 4., 2000, Manchester. **Proceedings** [...]. 2000. p. 29-39.

ZADROZNY, B.; ELKAN, C. Transforming classifier scores into accurate multiclass probability estimates. In: ACM SIGKDD INTERNATIONAL CONFERENCE ON KNOWLEDGE DISCOVERY AND DATA MINING, 8., 2002, Edmonton. **Proceedings** [...]. New York: ACM, 2002. p. 694-699.
