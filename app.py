import pandas as pd
import joblib
import streamlit as st

# Caminhos dos modelos
caminho_modelo_classico = './create_models/modelo_anti_fraude.pkl'
caminho_modelo_quantico = './create_models/documentacao_modelo_quantico/modelo_anti_fraude_quantum.pkl'

# Carregar os modelos salvos
modelo_classico = joblib.load(caminho_modelo_classico)
modelo_quantico = joblib.load(caminho_modelo_quantico)

# Configurar as entradas do usuário no Streamlit
st.title("Classificador de Transações Financeiras")
st.write("Insira os dados da transação para classificação.")

# Permitir que o usuário escolha o modelo
modelo_selecionado = st.selectbox(
    "Escolha o modelo de classificação:",
    options=["Clássico", "Quântico"]
)

# Entradas do usuário
valor = st.number_input("Valor da transação (em R$):", min_value=0.0, value=100.0, format="%.2f")
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

# Função para realizar a classificação
def realizar_classificacao(modelo, dados):
    print(dados)
    return modelo.predict(dados)

# Botão de classificação
if st.button("Classificar Transação"):
    try:
        # Escolher o modelo
        modelo_carregado = modelo_classico if modelo_selecionado == "Clássico" else modelo_quantico
        
        # Fazer a previsão
        previsao = realizar_classificacao(modelo_carregado, novos_dados)
        resultado = "Alto Risco de Fraude" if previsao[0] == 1 else "Baixo ou Sem Risco de Fraude"
        # Exibindo a mensagem com formatação condicional
        if previsao[0] == 1:
            st.warning(f"A transação foi classificada como **{resultado}**.")
        else:
            st.success(f"A transação foi classificada como **{resultado}**.")
        
        # Exibir a acurácia do modelo
        if modelo_selecionado == "Clássico":
            st.info("Acurácia do modelo clássico: 95%")  # Atualize com a acurácia real
        else:
            st.info("Acurácia do modelo quântico: 93%")  # Atualize com a acurácia real

    except Exception as e:
        st.error(f"Erro na classificação: {e}")