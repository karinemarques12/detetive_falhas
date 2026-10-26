import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Detetive de Falhas na Mineração", page_icon="⛏️", layout="wide")

st.title("⛏️ Detetive de Falhas na Mineração")
st.write(
    "Sistema acadêmico demonstrativo que usa uma Árvore de Decisão para classificar "
    "o nível de atenção recomendado para um equipamento."
)
st.warning(
    "Demonstração educativa: os dados e os resultados não substituem inspeção técnica, "
    "manuais do fabricante ou procedimentos de segurança. O modelo inicial é treinado "
    "com dados simulados, não com falhas reais."
)

# Valores categóricos usados para transformar as respostas em números.
opcoes = {
    "temperatura": ["Normal", "Acima do normal", "Muito alta"],
    "vibracao": ["Normal", "Um pouco elevada", "Muito elevada"],
    "ruido": ["Não", "Sim, leves", "Sim, fortes"],
    "oleo": ["Normal", "Abaixo do recomendado", "Contaminado ou com aspecto incomum", "Não verificado"],
    "vazamento": ["Não", "Sim, pequenos", "Sim, significativos"],
    "horas": ["Até 4 horas", "4 a 8 horas", "8 a 12 horas", "Mais de 12 horas"],
    "falhas": ["Nenhuma", "Uma falha recente", "Duas ou mais falhas"],
    "manutencao": ["Em dia", "Próxima do vencimento", "Atrasada", "Não sei"],
    "consumo": ["Normal", "Acima do habitual", "Muito acima do habitual", "Não disponível"],
    "producao": ["Normal", "Abaixo do esperado", "Muito abaixo do esperado", "Não se aplica"],
    "paradas": ["Nenhuma", "Uma parada", "Duas ou mais paradas"],
    "alertas": ["Nenhum", "Alerta leve", "Alerta crítico", "Não verificado"],
}

colunas = list(opcoes.keys())
rotulos = ["Risco baixo", "Atenção", "Risco elevado"]

# Criamos dados sintéticos para demonstrar o treinamento.
# A função abaixo simula exemplos para fins didáticos; não representa uma regra técnica real.
def criar_dados_simulados(n=900, seed=42):
    import numpy as np
    rng = np.random.default_rng(seed)
    linhas = []
    for _ in range(n):
        linha = {c: rng.choice(vals) for c, vals in opcoes.items()}
        pontos = 0
        pontos += {"Normal": 0, "Acima do normal": 1, "Muito alta": 3}[linha["temperatura"]]
        pontos += {"Normal": 0, "Um pouco elevada": 1, "Muito elevada": 2}[linha["vibracao"]]
        pontos += {"Não": 0, "Sim, leves": 1, "Sim, fortes": 2}[linha["ruido"]]
        pontos += {"Normal": 0, "Abaixo do recomendado": 2,
                   "Contaminado ou com aspecto incomum": 3, "Não verificado": 1}[linha["oleo"]]
        pontos += {"Não": 0, "Sim, pequenos": 1, "Sim, significativos": 3}[linha["vazamento"]]
        pontos += {"Até 4 horas": 0, "4 a 8 horas": 0, "8 a 12 horas": 1, "Mais de 12 horas": 2}[linha["horas"]]
        pontos += {"Nenhuma": 0, "Uma falha recente": 1, "Duas ou mais falhas": 2}[linha["falhas"]]
        pontos += {"Em dia": 0, "Próxima do vencimento": 1, "Atrasada": 2, "Não sei": 1}[linha["manutencao"]]
        pontos += {"Normal": 0, "Acima do habitual": 1, "Muito acima do habitual": 2, "Não disponível": 0}[linha["consumo"]]
        pontos += {"Normal": 0, "Abaixo do esperado": 1, "Muito abaixo do esperado": 2, "Não se aplica": 0}[linha["producao"]]
        pontos += {"Nenhuma": 0, "Uma parada": 1, "Duas ou mais paradas": 2}[linha["paradas"]]
        pontos += {"Nenhum": 0, "Alerta leve": 1, "Alerta crítico": 4, "Não verificado": 1}[linha["alertas"]]
        # Ruído aleatório pequeno para evitar uma correspondência perfeitamente determinística.
        pontos += int(rng.choice([0, 0, 0, 1]))
        if (linha["temperatura"] == "Muito alta" or
            linha["vibracao"] == "Muito elevada" or
            linha["oleo"] == "Contaminado ou com aspecto incomum" or
            linha["vazamento"] == "Sim, significativos" or
            linha["alertas"] == "Alerta crítico"):
            classe = "Risco elevado"
        elif pontos >= 9:
            classe = "Risco elevado"
        elif pontos >= 4:
            classe = "Atenção"
        else:
            classe = "Risco baixo"
        linhas.append({**linha, "resultado": classe})
    return pd.DataFrame(linhas)

dados = criar_dados_simulados(n=3000)
X = pd.get_dummies(dados[colunas])
y = dados["resultado"]
# A estratificação só funciona se cada classe tiver pelo menos 2 exemplos.
# Com dados simulados, uma classe pode aparecer muito poucas vezes.
contagens_classes = y.value_counts()
estratificar = y if len(contagens_classes) > 1 and contagens_classes.min() >= 2 else None

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=estratificar
)
modelo = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced")
modelo.fit(X_treino, y_treino)
acuracia = accuracy_score(y_teste, modelo.predict(X_teste))

st.sidebar.header("Sobre o modelo")
st.sidebar.metric("Acurácia nos dados simulados", f"{acuracia:.1%}")
st.sidebar.write("Profundidade máxima da árvore:", modelo.get_depth())
st.sidebar.caption("Essa métrica mede o desempenho em exemplos simulados. Não representa a precisão em equipamentos reais.")

st.subheader("1. Preencha a ficha do equipamento")
with st.form("formulario_diagnostico"):
    tipo = st.selectbox("Tipo de equipamento", [
        "Caminhão fora de estrada", "Escavadeira", "Pá carregadeira",
        "Trator de esteira", "Perfuratriz"
    ])
    c1, c2 = st.columns(2)
    respostas = {}
    for i, campo in enumerate(colunas):
        destino = c1 if i % 2 == 0 else c2
        with destino:
            nomes = {
                "temperatura": "Temperatura do equipamento",
                "vibracao": "Nível de vibração",
                "ruido": "Ruídos incomuns",
                "oleo": "Nível/condição do óleo",
                "vazamento": "Vazamentos visíveis",
                "horas": "Horas de operação por dia",
                "falhas": "Histórico recente de falhas",
                "manutencao": "Manutenção preventiva",
                "consumo": "Consumo de combustível/energia",
                "producao": "Desempenho da produção",
                "paradas": "Paradas inesperadas recentes",
                "alertas": "Alertas no painel",
            }
            respostas[campo] = st.selectbox(nomes[campo], opcoes[campo], key=f"campo_{campo}")
    enviado = st.form_submit_button("Analisar equipamento", type="primary")

if enviado:
    nova_linha = pd.DataFrame([respostas])
    entrada = pd.get_dummies(nova_linha)
    entrada = entrada.reindex(columns=X.columns, fill_value=0)
    previsao = modelo.predict(entrada)[0]
    probabilidades = modelo.predict_proba(entrada)[0]
    classes_modelo = list(modelo.classes_)
    prob = float(probabilidades[classes_modelo.index(previsao)])

    st.subheader("2. Resultado da análise")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Classificação prevista", previsao)
    with col_b:
        st.metric("Confiança estimada pelo modelo", f"{prob:.1%}")
    st.caption("A confiança é a probabilidade estimada pelo modelo treinado em dados simulados; não é garantia de diagnóstico correto.")

    fatores = []
    if respostas["temperatura"] != "Normal": fatores.append(f"Temperatura: {respostas['temperatura']}")
    if respostas["vibracao"] != "Normal": fatores.append(f"Vibração: {respostas['vibracao']}")
    if respostas["ruido"] != "Não": fatores.append(f"Ruídos: {respostas['ruido']}")
    if respostas["oleo"] != "Normal": fatores.append(f"Óleo: {respostas['oleo']}")
    if respostas["vazamento"] != "Não": fatores.append(f"Vazamento: {respostas['vazamento']}")
    if respostas["falhas"] != "Nenhuma": fatores.append(f"Histórico de falhas: {respostas['falhas']}")
    if respostas["manutencao"] == "Atrasada": fatores.append("Manutenção preventiva atrasada")
    if respostas["consumo"] in ["Acima do habitual", "Muito acima do habitual"]: fatores.append(f"Consumo: {respostas['consumo']}")
    if respostas["producao"] in ["Abaixo do esperado", "Muito abaixo do esperado"]: fatores.append(f"Produção: {respostas['producao']}")
    if respostas["paradas"] != "Nenhuma": fatores.append(f"Paradas: {respostas['paradas']}")
    if respostas["alertas"] != "Nenhum": fatores.append(f"Alertas: {respostas['alertas']}")

    st.markdown("**Fatores de atenção informados**")
    if fatores:
        for fator in fatores:
            st.write("•", fator)
    else:
        st.write("Nenhum sinal de alerta foi informado na ficha.")

    st.markdown("**Próximas ações sugeridas**")
    if previsao == "Risco elevado":
        st.error("Priorize a segurança. Siga os procedimentos da empresa e solicite avaliação da equipe de manutenção antes de continuar a operação quando houver condição insegura.")
    elif previsao == "Atenção":
        st.warning("Consulte o histórico do equipamento e programe uma inspeção técnica conforme os procedimentos da empresa.")
    else:
        st.success("Mantenha as inspeções preventivas e acompanhe os indicadores. Risco baixo não significa ausência de falhas.")

    # Guarda a análise para permitir baixar um registro.
    registro = {"tipo_equipamento": tipo, **respostas, "classificacao_modelo": previsao,
                "confianca_estimada_simulada": round(prob, 4)}
    st.download_button(
        "Baixar resultado em CSV",
        data=pd.DataFrame([registro]).to_csv(index=False).encode("utf-8-sig"),
        file_name="resultado_diagnostico.csv",
        mime="text/csv"
    )

st.divider()
st.subheader("3. Visualize a Árvore de Decisão")
st.write("O gráfico mostra exemplos das perguntas que o modelo aprendeu a usar nos dados simulados.")
fig, ax = plt.subplots(figsize=(20, 9))
plot_tree(
    modelo, feature_names=list(X.columns), class_names=list(modelo.classes_),
    filled=True, rounded=True, max_depth=3, fontsize=7, ax=ax
)
st.pyplot(fig)
plt.close(fig)

st.subheader("4. Dados demonstrativos")
st.write("A tabela abaixo mostra algumas linhas do conjunto simulado usado para treinar o modelo.")
st.dataframe(dados.head(10), use_container_width=True)
st.download_button(
    "Baixar conjunto de dados simulado",
    data=dados.to_csv(index=False).encode("utf-8-sig"),
    file_name="dados_simulados_equipamentos.csv",
    mime="text/csv"
)

st.info(
    "Para transformar este protótipo em uma solução confiável, substitua os dados simulados por "
    "registros reais e revisados por profissionais de manutenção, defina rótulos consistentes e "
    "avalie o modelo em dados que não participaram do treinamento."
)
