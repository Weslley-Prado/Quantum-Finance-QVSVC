# import streamlit as st
# import numpy as np
# import joblib

# # Carregamento do modelo
# svc_classical = joblib.load('classical_model.pkl')

# quantum_model = joblib.load('quantum_model.pkl')

# # Configuração da página
# st.set_page_config(page_title="Quantum Finance QVSVC", page_icon="💰", layout="centered")

# # Cabeçalho
# st.markdown("<h1 style='text-align: center; color: #4B0082;'>Quantum Finance SVM</h1>", unsafe_allow_html=True)
# st.markdown("<h4 style='text-align: center; color: #6A5ACD;'>Análise de Transações Financeiras com Machine Learning</h4>", unsafe_allow_html=True)
# st.write("---")

# # Entrada de dados
# st.subheader("Insira as características da transação:")
# valor_reais = st.number_input("Valor da Transação (em R$)", min_value=0.0, value=100.0, format="%.2f")
# tempo_horas = st.slider("Tempo da Transação (em horas)", min_value=0.0, max_value=24.0, value=12.0)
# fim_de_semana = st.selectbox("A transação ocorreu no final de semana?", options=[0, 1], format_func=lambda x: "Sim" if x else "Não")
# idade_cliente = st.slider("Idade do Cliente", min_value=18, max_value=100, value=30)
# latitude = st.number_input("Latitude da Localização", min_value=-30.0, max_value=-15.0, value=-20.0)
# longitude = st.number_input("Longitude da Localização", min_value=-60.0, max_value=-35.0, value=-47.0)

# # Normalizações (compatíveis com o modelo)
# valor_normalizado = valor_reais / 10000  # Normalizado com base no intervalo [0, 10.000]
# tempo_normalizado = tempo_horas / 24  # Normalizado para [0, 1]

# # Construir entrada para o modelo
# entrada = np.array([[valor_normalizado, tempo_normalizado, fim_de_semana, idade_cliente, latitude, longitude]])

# # Previsão com SVM Clássico
# if st.button("Classificar"):
#     pred_classico = svc_classical.predict(entrada)
#     resultado_classico = "Fraudulenta" if pred_classico[0] == 1 else "Não Fraudulenta"
#     st.success(f"Resultado da Classificação: **{resultado_classico}**")

# if st.button("Classificar - QSVC"):
#     pred_quantum = quantum_model.predict(entrada)
#     resultado_classico = "Fraudulenta" if pred_quantum[0] == 1 else "Não Fraudulenta"
#     st.success(f"Resultado da Classificação: **{resultado_classico}**")

# # Explicação e rodapé
# st.write("---")
# st.write("""
# ### Informações:
# - **Valor da Transação**: Representa o valor da transação em reais (ex.: R$10.000).
# - **Tempo da Transação**: Representa a hora do dia em que a transação ocorreu (0-24h).
# - **Fim de Semana**: Indica se a transação ocorreu em um sábado ou domingo.
# - **Idade do Cliente**: Faixa etária do cliente.
# - **Latitude e Longitude**: Coordenadas da localização onde a transação ocorreu.

# As entradas são normalizadas internamente para compatibilidade com os modelos treinados.
# """)

# st.markdown(
#     """
#     <div style='text-align: center; color: gray;'>
#         Desenvolvido por Weslley Rosa Prado<br>
#         Orientador: Professor Doutor José Alexandre Nogueira<br>
#         Projeto Prático Baseado no Trabalho de Conclusão de Curso de Física<br>
#         Universidade Federal do Espírito Santo - UFES
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# import streamlit as st
# import numpy as np
# import joblib
# import pandas as pd

# # Carregar o modelo
# svc_classical = joblib.load('classical_model_com_perfis.pkl')

# # Configuração da página
# st.set_page_config(page_title="Quantum Finance QVSVC", page_icon="💰", layout="centered")

# # Cabeçalho
# st.markdown("<h1 style='text-align: center; color: #4B0082;'>Quantum Finance SVM</h1>", unsafe_allow_html=True)
# st.markdown("<h4 style='text-align: center; color: #6A5ACD;'>Análise de Transações Financeiras com Machine Learning</h4>", unsafe_allow_html=True)
# st.write("---")

# # Mapeamento de capitais para suas coordenadas
# capitais = {
#     "Rio Branco": (-9.0, -67.8),
#     "Maceió": (-9.6, -35.7),
#     "Macapá": (0.0, -51.0),
#     "Manaus": (-3.1, -60.0),
#     "Salvador": (-12.97, -38.51),
#     "Fortaleza": (-3.7, -38.5),
#     "Brasília": (-15.78, -47.93),
#     "Vitória": (-20.32, -40.3),
#     "Goiânia": (-16.68, -49.3),
#     "São Luís": (-2.55, -44.3),
#     "Cuiabá": (-15.6, -56.1),
#     "Campo Grande": (-20.5, -54.6),
#     "Belo Horizonte": (-19.92, -43.94),
#     "Belém": (-1.46, -48.49),
#     "João Pessoa": (-7.12, -34.88),
#     "Curitiba": (-25.43, -49.27),
#     "Recife": (-8.05, -34.9),
#     "Teresina": (-5.09, -42.81),
#     "Rio de Janeiro": (-22.91, -43.17),
#     "Natal": (-5.79, -35.2),
#     "Porto Alegre": (-30.03, -51.22),
#     "Aracaju": (-10.95, -37.07),
#     "Palmas": (-10.2, -48.3),
#     "São Paulo": (-23.55, -46.63),
#     "Porto Velho": (-8.76, -63.9),
#     "Boa Vista": (2.82, -60.7),
#     "Florianópolis": (-27.59, -48.55)
# }

# # Entrada de dados
# st.subheader("Insira as características da transação:")
# valor_reais = st.number_input("Valor da Transação (em R$)", min_value=0.0, value=100.0, format="%.2f")
# tempo_horas = st.slider("Tempo da Transação (em horas)", min_value=0.0, max_value=24.0, value=12.0)
# fim_de_semana = st.selectbox("A transação ocorreu no final de semana?", options=[0, 1], format_func=lambda x: "Sim" if x else "Não")
# idade_cliente = st.slider("Idade do Cliente", min_value=18, max_value=100, value=30)

# # Seleção da cidade (capital brasileira)
# cidade = st.selectbox("Selecione a Capital", options=list(capitais.keys()))

# # Obter a latitude e longitude da cidade selecionada
# latitude, longitude = capitais[cidade]

# # Normalizações (compatíveis com o modelo)
# valor_normalizado = valor_reais / 10000  # Normalizado com base no intervalo [0, 10.000]
# tempo_normalizado = tempo_horas / 24  # Normalizado para [0, 1]

# # Perfil do cliente (apenas exemplo, adicionar conforme necessário)
# perfil = st.selectbox("Selecione o Perfil do Cliente", options=['Aposentado', 'Estudante', 'Engenheiro'])

# # Codificação One-Hot para perfil
# perfils_possiveis = ['Aposentado', 'Estudante', 'Engenheiro']  # Exemplo de perfis
# perfil_encoded = [1 if perfil == p else 0 for p in perfils_possiveis]

# # Codificação One-Hot para cidade
# cidades_possiveis = list(capitais.keys())  # Exemplo de cidades
# cidade_encoded = [1 if cidade == c else 0 for c in cidades_possiveis]

# # Criar dataframe para verificar as colunas One-Hot (para garantir que as colunas estejam no formato correto)
# input_df = pd.DataFrame([perfil_encoded + cidade_encoded], columns=perfils_possiveis + cidades_possiveis)

# # Verificar a correspondência com o número de características do modelo
# expected_features = len(input_df.columns) + 6  # 6 outras variáveis além das One-Hot
# entrada = np.array([[valor_normalizado, tempo_normalizado, fim_de_semana, idade_cliente, latitude, longitude] + input_df.values.flatten().tolist()])

# # Previsão com SVM Clássico
# if st.button("Classificar"):
#     if entrada.shape[1] == expected_features:
#         pred_classico = svc_classical.predict(entrada)
#         resultado_classico = "Fraudulenta" if pred_classico[0] == 1 else "Não Fraudulenta"
#         st.success(f"Resultado da Classificação: **{resultado_classico}**")
#     else:
#         st.error("Erro: O número de características fornecidas não corresponde ao número esperado pelo modelo.")

# # Explicação e rodapé
# st.write("---")
# st.write("""
# ### Informações:
# - **Valor da Transação**: Representa o valor da transação em reais (ex.: R$10.000).
# - **Tempo da Transação**: Representa a hora do dia em que a transação ocorreu (0-24h).
# - **Fim de Semana**: Indica se a transação ocorreu em um sábado ou domingo.
# - **Idade do Cliente**: Faixa etária do cliente.
# - **Cidade**: Selecione a cidade da transação para obter as coordenadas de latitude e longitude.

# As entradas são normalizadas internamente para compatibilidade com os modelos treinados.
# """)

# st.markdown(
#     """
#     <div style='text-align: center; color: gray;'>
#         Desenvolvido por Weslley Rosa Prado<br>
#         Orientador: Professor Doutor José Alexandre Nogueira<br>
#         Projeto Prático Baseado no Trabalho de Conclusão de Curso de Física<br>
#         Universidade Federal do Espírito Santo - UFES
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# import streamlit as st
# import joblib
# import pandas as pd
# import numpy as np

# # Carregar os modelos, scaler e colunas de treinamento
# svc_classical = joblib.load('classical_model_com_perfis.pkl')
# scaler = joblib.load('scaler.pkl')
# expected_columns = joblib.load('training_columns.pkl')

# # Dados auxiliares
# capitais = {
#     "São Paulo": (-23.55, -46.63), "Rio de Janeiro": (-22.91, -43.17), "Belo Horizonte": (-19.92, -43.94),
#     "Brasília": (-15.78, -47.93), "Salvador": (-12.97, -38.51), "Fortaleza": (-3.7, -38.5),
#     "Curitiba": (-25.43, -49.27), "Manaus": (-3.1, -60.0), "Recife": (-8.05, -34.9),
#     "Porto Alegre": (-30.03, -51.22), "Vitória": (-20.32, -40.3), "Goiânia": (-16.68, -49.3),
#     "Belém": (-1.46, -48.49), "São Luís": (-2.55, -44.3), "Maceió": (-9.6, -35.7),
#     "Natal": (-5.79, -35.2), "Campo Grande": (-20.5, -54.6), "João Pessoa": (-7.12, -34.88),
#     "Aracaju": (-10.95, -37.07), "Teresina": (-5.09, -42.81), "Cuiabá": (-15.6, -56.1),
#     "Macapá": (0.0, -51.0), "Rio Branco": (-9.0, -67.8), "Boa Vista": (2.82, -60.7),
#     "Palmas": (-10.2, -48.3), "Florianópolis": (-27.59, -48.55), "Porto Velho": (-8.76, -63.9)
# }
# perfis = ["Aposentado", "Estudante", "Engenheiro", "Bancário", "Investidor", "Comerciante"]

# # Configuração do Streamlit
# st.set_page_config(page_title="Classificador de Transações", page_icon="💳", layout="centered")

# st.title("Classificador de Transações Financeiras")
# st.write("Insira os dados da transação para classificação.")

# # Entradas do usuário
# valor = st.number_input("Valor da transação (em R$):", min_value=0.0, value=100.0, format="%.2f")
# tempo = st.slider("Hora da transação (0-24h):", min_value=0.0, max_value=24.0, value=12.0)
# fim_de_semana = st.selectbox("A transação ocorreu no final de semana?", options=[0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
# idade = st.slider("Idade do cliente:", min_value=18, max_value=80, value=30)
# perfil = st.selectbox("Perfil do cliente:", options=perfis)
# cidade = st.selectbox("Cidade da transação:", options=list(capitais.keys()))

# # Obter latitude e longitude
# latitude, longitude = capitais[cidade]

# # Normalizar os dados contínuos
# dados_norm = scaler.transform([[valor, tempo, latitude, longitude]])
# valor_norm, tempo_norm, lat_norm, long_norm = dados_norm[0]

# # Codificação one-hot de variáveis categóricas
# perfil_encoded = [1 if p == perfil else 0 for p in perfis]
# cidade_encoded = [1 if c == cidade else 0 for c in capitais.keys()]

# # Criar DataFrame de entrada
# entrada = pd.DataFrame(
#     [[valor_norm, tempo_norm, fim_de_semana, idade, lat_norm, long_norm] + perfil_encoded + cidade_encoded],
#     columns=expected_columns
# )

# # Verifique as colunas do DataFrame 'entrada' e 'expected_columns'
# print("Colunas no DataFrame de entrada:", entrada.columns)
# print("Colunas esperadas pelo modelo:", expected_columns)

# # Alinhar as colunas de 'entrada' com 'expected_columns'
# entrada = entrada.reindex(columns=expected_columns, fill_value=0)

# # Verifique novamente
# print("Colunas após reindexação:", entrada.columns)

# # Classificação
# if st.button("Classificar"):
#     try:
#         predicao = svc_classical.predict(entrada)
#         resultado = "Fraudulenta" if predicao[0] == 1 else "Não Fraudulenta"
#         st.success(f"A transação foi classificada como **{resultado}**.")
#     except Exception as e:
#         st.error(f"Erro na classificação: {e}")

# # Informações adicionais
# st.write("---")
# st.markdown("""
# #### Sobre:
# - Este classificador usa um modelo SVM treinado com dados simulados.
# - As variáveis categóricas como perfil e cidade foram codificadas com One-Hot Encoding.
# - As variáveis contínuas foram normalizadas para compatibilidade com o modelo.
# """)


# import streamlit as st
# import joblib
# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder

# Carregar o modelo salvo
# modelo_carregado = joblib.load('./create_models/modelo_anti_fraude.pkl')

# # Dados auxiliares para a interface
# capitais = {
#     "São Paulo": (-23.55, -46.63), "Rio de Janeiro": (-22.91, -43.17), "Belo Horizonte": (-19.92, -43.94),
#     "Brasília": (-15.78, -47.93), "Salvador": (-12.97, -38.51), "Fortaleza": (-3.7, -38.5),
#     "Curitiba": (-25.43, -49.27), "Manaus": (-3.1, -60.0), "Recife": (-8.05, -34.9),
#     "Porto Alegre": (-30.03, -51.22), "Vitória": (-20.32, -40.3), "Goiânia": (-16.68, -49.3),
#     "Belém": (-1.46, -48.49), "São Luís": (-2.55, -44.3), "Maceió": (-9.6, -35.7),
#     "Natal": (-5.79, -35.2), "Campo Grande": (-20.5, -54.6), "João Pessoa": (-7.12, -34.88),
#     "Aracaju": (-10.95, -37.07), "Teresina": (-5.09, -42.81), "Cuiabá": (-15.6, -56.1),
#     "Macapá": (0.0, -51.0), "Rio Branco": (-9.0, -67.8), "Boa Vista": (2.82, -60.7),
#     "Palmas": (-10.2, -48.3), "Florianópolis": (-27.59, -48.55), "Porto Velho": (-8.76, -63.9)
# }
# perfis = ["Aposentado", "Estudante", "Engenheiro", "Bancário", "Investidor", "Comerciante"]

# # Configuração do Streamlit
# st.set_page_config(page_title="Classificador de Transações", page_icon="💳", layout="centered")

# st.title("Classificador de Transações Financeiras")
# st.write("Insira os dados da transação para classificação.")

# # Entradas do usuário
# valor = st.number_input("Valor da transação (em R$):", min_value=0.0, value=100.0, format="%.2f")
# tempo = st.slider("Hora da transação (0-24h):", min_value=0.0, max_value=24.0, value=12.0)
# fim_de_semana = st.selectbox("A transação ocorreu no final de semana?", options=[0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
# idade = st.slider("Idade do cliente:", min_value=18, max_value=80, value=30)
# perfil = st.selectbox("Perfil do cliente:", options=perfis)
# cidade = st.selectbox("Cidade da transação:", options=list(capitais.keys()))
# tipo_transacao = st.selectbox("Tipo de transação:", options=["compra", "saque", "transferência"])  # Exemplo de transações
# numero_transacoes = st.slider("Número de transações do cliente:", min_value=1, max_value=20, value=5)

# # Obter latitude e longitude
# latitude, longitude = capitais[cidade]

# # Criar lista com as entradas
# novos_dados = [[valor, tempo, idade, latitude, longitude, numero_transacoes, tipo_transacao, fim_de_semana ]]

# # Aplicar One-Hot Encoding para 'perfil' e 'cidade'
# encoder_perfil = OneHotEncoder(sparse_output=False)  # Correção: sparse_output=False
# encoder_cidade = OneHotEncoder(sparse_output=False)  # Correção: sparse_output=False

# # Transformar as variáveis categóricas 'perfil' e 'cidade'
# perfil_encoded = encoder_perfil.fit_transform([[perfil]])
# cidade_encoded = encoder_cidade.fit_transform([[cidade]])

# # Combinar todas as entradas
# entrada_completa = novos_dados[0][:6] + perfil_encoded[0].tolist() + cidade_encoded[0].tolist() + novos_dados[0][6:]

# # Exibir a entrada completa para conferência
# st.write(f"Entrada para o modelo: {entrada_completa}")

# # Realizar a previsão
# if st.button("Classificar Transação"):
#     try:
#         print(entrada_completa)
#         # Fazer a previsão
#         previsao = modelo_carregado.predict([entrada_completa])
#         resultado = "Fraudulenta" if previsao[0] == 1 else "Não Fraudulenta"
#         st.success(f"A transação foi classificada como **{resultado}**.")
#     except Exception as e:
#         st.error(f"Erro na classificação: {e}")

# # Informações adicionais sobre o modelo
# st.write("---")
# st.markdown("""
# #### Sobre:
# - Este classificador usa um modelo SVM treinado com dados simulados para detectar transações fraudulentas.
# - As variáveis categóricas como perfil e cidade são processadas diretamente no modelo.
# - Para melhorar a precisão, as variáveis contínuas, como valor e hora da transação, foram consideradas no treinamento do modelo.
# """)


import pandas as pd
import joblib
import streamlit as st

# Carregar o modelo salvo

# Carregar o modelo salvo
modelo_carregado = joblib.load('./create_models/modelo_anti_fraude.pkl')
# Configurar as entradas do usuário no Streamlit
st.title("Classificador de Transações Financeiras")
st.write("Insira os dados da transação para classificação.")

# Entradas do usuário
valor = st.number_input("Valor da transação (em R$):", min_value=0.0, value=100.0, format="%.2f")
# tempo = st.slider("Hora da transação (0-24h):", min_value=0.0, max_value=24.0, value=12.0)
# Slider de tempo em intervalos de 30 minutos
tempo = st.slider(
    "Hora da transação (0-24h):",
    min_value=0.0,  # Início do intervalo
    max_value=24.0, # Fim do intervalo
    value=12.0,     # Valor inicial
    step=0.5        # Incremento de 30 minutos
)
fim_de_semana = st.selectbox("A transação ocorreu no final de semana?", options=[0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
idade_cliente = st.slider("Idade do cliente:", min_value=18, max_value=80, value=30)
numero_transacoes = st.number_input("Número de transações no dia:", min_value=1, value=5)
tipo_transacao = st.selectbox("Tipo de transação:", options=["Compra", "Saque", "Transferência"])
cidade = st.selectbox("Cidade da transação:", options=[
    "São Paulo", "Rio de Janeiro", "Belo Horizonte", "Brasília", "Salvador", 
    "Fortaleza", "Curitiba", "Manaus", "Recife", "Porto Alegre", 
    "Vitória", "Goiânia", "Belém", "São Luís", "Maceió", 
    "Natal", "Campo Grande", "João Pessoa", "Aracaju", "Teresina", 
    "Cuiabá", "Macapá", "Rio Branco", "Boa Vista", "Palmas", 
    "Florianópolis", "Porto Velho"
])
perfil = st.selectbox("Perfil do cliente:", options=["Aposentado", "Estudante", "Não Aposentado"])

# Criar DataFrame com as entradas
novos_dados = pd.DataFrame([{
    "valor": valor,
    "tempo": tempo,
    "fim_de_semana": fim_de_semana,
    "idade_cliente": idade_cliente,
    "numero_transacoes": numero_transacoes,
    "tipo_transacao": tipo_transacao,
    "cidade": cidade,
    "perfil": perfil
}])

# Realizar a classificação
if st.button("Classificar Transação"):
    try:
        previsao = modelo_carregado.predict(novos_dados)
        resultado = "Fraudulenta" if previsao[0] == 1 else "Não Fraudulenta"
        st.success(f"A transação foi classificada como **{resultado}**.")
    except Exception as e:
        st.error(f"Erro na classificação: {e}")


print(valor*numero_transacoes)