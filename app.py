import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Detetive de Falhas na Mineração", page_icon="⛏️", layout="wide")
st.title("⛏️ Detetive de Falhas na Mineração")
st.write("Preencha as condições do equipamento para estimar o nível de risco e visualizar como a árvore de decisão chegou à classificação.")
st.info("Demonstração educacional com dados simulados. Não substitui inspeção técnica, sensores calibrados ou procedimentos de segurança da mina.")

# Os dados abaixo são sintéticos e servem apenas para demonstrar o funcionamento do ML.
def criar_dados_simulados(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    tipos = ["Caminhão fora de estrada", "Escavadeira", "Britador", "Correia transportadora", "Perfuratriz"]
    categorias = {
        "temperatura": ["Normal", "Acima do normal", "Muito alta"],
        "vibracao": ["Normal", "Moderada", "Alta"],
        "ruido": ["Normal", "Incomum", "Muito alto"],
        "oleo": ["Bom", "Atenção", "Degradado"],
        "vazamento": ["Não", "Pequeno", "Grande"],
        "falhas": ["Nenhuma", "Uma falha recente", "Falhas recorrentes"],
        "manutencao": ["Em dia", "Atrasada", "Muito atrasada"],
        "consumo": ["Normal", "Acima do normal", "Muito alto"],
        "producao": ["Normal", "Reduzida", "Muito reduzida"],
        "paradas": ["Não", "Ocasional", "Frequentes"],
        "alertas": ["Nenhum", "Alguns alertas", "Alertas críticos"],
    }
    pesos = {
        "temperatura": [0.55, 0.30, 0.15], "vibracao": [0.55, 0.30, 0.15],
        "ruido": [0.60, 0.27, 0.13], "oleo": [0.55, 0.27, 0.18],
        "vazamento": [0.70, 0.20, 0.10], "falhas": [0.60, 0.25, 0.15],
        "manutencao": [0.60, 0.25, 0.15], "consumo": [0.60, 0.27, 0.13],
        "producao": [0.60, 0.27, 0.13], "paradas": [0.65, 0.23, 0.12],
        "alertas": [0.65, 0.23, 0.12],
    }
    dados = pd.DataFrame({"equipamento": rng.choice(tipos, n)})
    for coluna, opcoes in categorias.items():
        dados[coluna] = [rng.choice(opcoes, p=pesos[coluna]) for _ in range(n)]
    dados["horas"] = rng.integers(1, 25, n)

    pontos = np.zeros(n)
    pontos += dados["temperatura"].map({"Normal": 0, "Acima do normal": 1, "Muito alta": 2}).to_numpy()
    pontos += dados["vibracao"].map({"Normal": 0, "Moderada": 1, "Alta": 2}).to_numpy()
    pontos += dados["ruido"].map({"Normal": 0, "Incomum": 1, "Muito alto": 2}).to_numpy()
    pontos += dados["oleo"].map({"Bom": 0, "Atenção": 1, "Degradado": 2}).to_numpy()
    pontos += dados["vazamento"].map({"Não": 0, "Pequeno": 1, "Grande": 2}).to_numpy()
    pontos += dados["falhas"].map({"Nenhuma": 0, "Uma falha recente": 1, "Falhas recorrentes": 2}).to_numpy() * 1.2
    pontos += dados["manutencao"].map({"Em dia": 0, "Atrasada": 1, "Muito atrasada": 2}).to_numpy()
    pontos += dados["consumo"].map({"Normal": 0, "Acima do normal": 1, "Muito alto": 2}).to_numpy()
    pontos += dados["producao"].map({"Normal": 0, "Reduzida": 1, "Muito reduzida": 2}).to_numpy()
    pontos += dados["paradas"].map({"Não": 0, "Ocasional": 1, "Frequentes": 2}).to_numpy()
    pontos += dados["alertas"].map({"Nenhum": 0, "Alguns alertas": 1, "Alertas críticos": 2}).to_numpy() * 1.2
    pontos += np.where(dados["horas"].to_numpy() >= 20, 2, np.where(dados["horas"].to_numpy() >= 12, 1, 0))
    dados["resultado"] = np.where(pontos >= 13, "Risco elevado", np.where(pontos >= 7, "Atenção", "Risco baixo"))
    return dados

DADOS = criar_dados_simulados()
FEATURES = ["equipamento", "temperatura", "vibracao", "ruido", "oleo", "vazamento", "horas", "falhas", "manutencao", "consumo", "producao", "paradas", "alertas"]
X = pd.get_dummies(DADOS[FEATURES])
y = DADOS["resultado"]
# Evita erro de stratify caso alguma classe tenha poucos exemplos em alterações futuras.
contagens = y.value_counts()
estratificar = y if len(contagens) > 1 and contagens.min() >= 2 else None
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.25, random_state=42, stratify=estratificar)
modelo = DecisionTreeClassifier(max_depth=5, min_samples_leaf=12, class_weight="balanced", random_state=42)
modelo.fit(X_treino, y_treino)

# Tradução dos nomes das variáveis codificadas para a árvore.
NOMES_PT = {
    "equipamento": "Tipo de equipamento", "temperatura": "Temperatura", "vibracao": "Vibração",
    "ruido": "Ruídos incomuns", "oleo": "Condição do óleo", "vazamento": "Vazamentos",
    "horas": "Horas de operação por dia", "falhas": "Histórico de falhas",
    "manutencao": "Manutenção preventiva", "consumo": "Consumo de combustível/energia",
    "producao": "Desempenho da produção", "paradas": "Paradas inesperadas", "alertas": "Alertas no painel",
}
def nome_coluna_pt(nome):
    if nome == "horas":
        return NOMES_PT["horas"]
    for prefixo, traducao in NOMES_PT.items():
        if nome.startswith(prefixo + "_"):
            return traducao + " — " + nome[len(prefixo) + 1:]
    return nome

with st.sidebar:
    st.header("Condição do equipamento")
    equipamento = st.selectbox("Tipo de equipamento", ["Caminhão fora de estrada", "Escavadeira", "Britador", "Correia transportadora", "Perfuratriz"])
    temperatura = st.selectbox("Temperatura", ["Normal", "Acima do normal", "Muito alta"])
    vibracao = st.selectbox("Vibração", ["Normal", "Moderada", "Alta"])
    ruido = st.selectbox("Ruídos incomuns", ["Normal", "Incomum", "Muito alto"])
    oleo = st.selectbox("Condição do óleo", ["Bom", "Atenção", "Degradado"])
    vazamento = st.selectbox("Vazamentos", ["Não", "Pequeno", "Grande"])
    horas = st.slider("Horas de operação por dia", 1, 24, 12)
    falhas = st.selectbox("Histórico de falhas", ["Nenhuma", "Uma falha recente", "Falhas recorrentes"])
    manutencao = st.selectbox("Manutenção preventiva", ["Em dia", "Atrasada", "Muito atrasada"])
    consumo = st.selectbox("Consumo de combustível/energia", ["Normal", "Acima do normal", "Muito alto"])
    producao = st.selectbox("Desempenho da produção", ["Normal", "Reduzida", "Muito reduzida"])
    paradas = st.selectbox("Paradas inesperadas", ["Não", "Ocasional", "Frequentes"])
    alertas = st.selectbox("Alertas no painel", ["Nenhum", "Alguns alertas", "Alertas críticos"])

respostas = {
    "equipamento": equipamento, "temperatura": temperatura, "vibracao": vibracao, "ruido": ruido,
    "oleo": oleo, "vazamento": vazamento, "horas": horas, "falhas": falhas,
    "manutencao": manutencao, "consumo": consumo, "producao": producao, "paradas": paradas, "alertas": alertas,
}
entrada_bruta = pd.DataFrame([respostas])[FEATURES]
entrada = pd.get_dummies(entrada_bruta).reindex(columns=X.columns, fill_value=0)
predicao = modelo.predict(entrada)[0]
probabilidades = modelo.predict_proba(entrada)[0]
confianca = float(probabilidades.max())

st.subheader("Resultado da avaliação")
col1, col2, col3 = st.columns(3)
col1.metric("Classificação estimada", predicao)
col2.metric("Confiança do modelo", f"{confianca:.0%}")
col3.metric("Acurácia de teste (dados simulados)", f"{accuracy_score(y_teste, modelo.predict(X_teste)):.0%}")

if predicao == "Risco elevado":
    st.error("Risco elevado: interrompa a avaliação operacional conforme os procedimentos locais e solicite inspeção da equipe responsável antes de continuar a operação.")
    recomendacoes = ["Solicitar inspeção técnica prioritária.", "Verificar alertas, vazamentos, temperatura e vibração.", "Consultar o histórico de falhas e os procedimentos de bloqueio/etiquetagem aplicáveis."]
elif predicao == "Atenção":
    st.warning("Atenção: planeje uma verificação técnica e acompanhe as condições do equipamento.")
    recomendacoes = ["Programar inspeção preventiva em prazo adequado.", "Monitorar os sinais que estão fora do padrão.", "Confirmar se a manutenção está em dia e registrar as observações."]
else:
    st.success("Risco baixo segundo o modelo demonstrativo. Continue seguindo as rotinas de inspeção e manutenção.")
    recomendacoes = ["Manter a manutenção preventiva programada.", "Continuar monitorando sensores e alertas.", "Registrar os dados para acompanhar tendências ao longo do tempo."]
st.markdown("**Recomendações**")
for item in recomendacoes:
    st.write(f"- {item}")

st.subheader("Por que o modelo chegou a esse resultado?")
st.caption("O caminho abaixo mostra as decisões da árvore para as respostas atuais. Em seguida, a árvore completa é exibida com os nomes em português.")
# decision_path retorna os nós atravessados pela amostra atual.
no_indicador = modelo.decision_path(entrada)
os_nos = no_indicador.indices
arvore = modelo.tree_
passos = []
for no in os_nos:
    if arvore.feature[no] < 0:
        continue
    indice_feature = arvore.feature[no]
    nome_original = X.columns[indice_feature]
    valor = float(entrada.iloc[0, indice_feature])
    limite = float(arvore.threshold[no])
    decisao = "sim" if valor <= limite else "não"
    nome_legivel = nome_coluna_pt(nome_original)
    if nome_original in entrada.columns and "_" in nome_original:
        valor_legivel = "presente" if valor == 1 else "ausente"
    else:
        valor_legivel = str(valor)
    passos.append(f"**{len(passos)+1}. {nome_legivel}**: valor codificado `{valor_legivel}`; condição `{valor:.0f} ≤ {limite:.2f}`? **{decisao.upper()}**")
if passos:
    for passo in passos:
        st.markdown(passo)
else:
    st.write("A árvore classificou a entrada diretamente em uma folha, sem decisões intermediárias.")

with st.expander("Ver árvore de decisão completa em português", expanded=True):
    fig, ax = plt.subplots(figsize=(32, 20))
    plot_tree(
        modelo,
        feature_names=[nome_coluna_pt(c) for c in X.columns],
        class_names=[str(c) for c in modelo.classes_],
        filled=True, rounded=True, max_depth=None, fontsize=7, ax=ax,
    )
    ax.set_title("Árvore de decisão — variáveis e classes em português")
    st.pyplot(fig, use_container_width=True)
    st.caption("Esta visualização exibe todos os níveis da árvore treinada. Se estiver muito grande, amplie a imagem ou use o zoom do navegador; o caminho individual acima destaca as decisões tomadas para as respostas atuais.")

with st.expander("Ver dados usados nesta demonstração"):
    st.write("Os dados foram gerados artificialmente para fins didáticos, usando regras simuladas. Não representam medições reais de uma mina.")
    st.dataframe(DADOS.head(20), use_container_width=True)
