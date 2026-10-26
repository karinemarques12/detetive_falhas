# Detetive de Falhas na Mineração

Projeto acadêmico demonstrativo de Árvore de Decisão com Python e Streamlit.

## 1. O que o projeto faz
- Exibe uma ficha com 13 perguntas sobre o equipamento.
- Treina uma Árvore de Decisão com exemplos simulados.
- Classifica o caso informado como `Risco baixo`, `Atenção` ou `Risco elevado`.
- Mostra fatores de atenção, sugestões gerais e o desenho da árvore.
- Permite baixar o resultado e os dados demonstrativos em CSV.

## 2. Requisitos
Instale Python 3.10 ou superior. Durante a instalação do Python no Windows, marque a opção **Add Python to PATH**.

## 3. Como executar no Windows
1. Extraia o arquivo ZIP para uma pasta.
2. Abra essa pasta no Explorador de Arquivos.
3. Clique na barra de endereço, digite `cmd` e pressione Enter.
4. Crie um ambiente virtual:

   ```bat
   python -m venv .venv
   ```

5. Ative-o:

   ```bat
   .venv\Scripts\activate
   ```

6. Instale as bibliotecas:

   ```bat
   pip install -r requirements.txt
   ```

7. Inicie a aplicação:

   ```bat
   streamlit run app.py
   ```

8. O Streamlit abrirá a aplicação no navegador. Se não abrir automaticamente, use o endereço local que aparecer no terminal (normalmente `http://localhost:8501`).

## 4. Como apresentar o projeto
1. Explique o problema: identificar sinais que merecem inspeção em equipamentos.
2. Preencha a ficha com um cenário normal.
3. Faça outro teste com sinais de alerta e compare os resultados.
4. Mostre a árvore de decisão e explique que o modelo percorre divisões aprendidas durante o treinamento.
5. Baixe o CSV e explique como os dados podem ser analisados.

## 5. Importante: dados simulados
O conjunto de treinamento é gerado no próprio código para permitir executar o projeto sem depender de dados externos. Os rótulos foram criados por regras didáticas; portanto, a acurácia mostrada só descreve o desempenho nesse conjunto simulado. Ela não prova capacidade de prever falhas reais.

Para um projeto real, reúna dados históricos confiáveis, obtenha rótulos validados por especialistas, remova dados pessoais ou confidenciais e teste o modelo em registros separados do treinamento. Não use esta demonstração para decidir se um equipamento real pode continuar operando.


### Árvore de decisão em português
Depois de preencher a ficha e clicar em **Analisar equipamento**, o aplicativo exibe uma versão resumida da árvore usada pelo modelo. Os nomes das variáveis são apresentados em português e o gráfico mostra apenas os primeiros níveis para facilitar a leitura. O modelo continua sendo treinado com dados simulados para fins acadêmicos.
