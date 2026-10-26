# Detetive de Falhas na Mineração

Aplicação educacional em Python e Streamlit para demonstrar classificação com Árvore de Decisão (Machine Learning).

## O que o aplicativo faz
- Recebe condições de um equipamento por meio de campos interativos.
- Classifica a situação em **Risco baixo**, **Atenção** ou **Risco elevado**.
- Mostra recomendações gerais de acompanhamento.
- Explica o caminho de decisões seguido pela árvore para as respostas atuais.
- Exibe a árvore de decisão com nomes de variáveis e classes em português.

## Como executar
1. Instale Python 3.10 ou superior.
2. Extraia o arquivo ZIP e abra um terminal dentro da pasta do projeto.
3. (Opcional, recomendado) Crie um ambiente virtual.
4. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

5. Inicie o aplicativo:

   ```bash
   streamlit run app.py
   ```

O Streamlit abrirá o aplicativo no navegador. Os campos na barra lateral atualizam a classificação quando são alterados.

## Importante
Os dados e as regras de classificação são **simulados**, exclusivamente para demonstração. A saída não deve ser usada para liberar equipamentos, substituir inspeções, orientar decisões operacionais reais ou substituir os procedimentos de segurança da mina. Para uso real, seria necessário treinar e validar o modelo com dados históricos confiáveis e avaliação de especialistas.
