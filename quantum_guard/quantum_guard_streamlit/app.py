import pandas as pd
import joblib
import streamlit as st

# Caminhos dos modelos
caminho_modelo_classico = './infra/model_of_classical_machine_learning/modelo_anti_fraude.pkl'
caminho_modelo_quantico = './infra/model_of_quantum_machine_learning/modelo_anti_fraude_quantum.pkl'

# Carregar os modelos salvos
modelo_classico = joblib.load(caminho_modelo_classico)
modelo_quantico = joblib.load(caminho_modelo_quantico)

# Configuração do estilo da página
st.set_page_config(page_title="Quantum Guard", page_icon="💰", layout="centered")
# Adicionar estilo CSS para limitar a altura e evitar rolagem


# Cabeçalho
st.markdown("<h1 style='font-size:24px; text-align: center; color: #6A5ACD; padding: 0rem'>Quantum Guard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='font-size:18px; text-align: center; color: #6A5ACD;'>Classificador de Transações Financeiras de Transações Financeiras com Machine Learning Clássico e Quântico</h4>", unsafe_allow_html=True)

modelo_selecionado = st.selectbox(
        "Escolha o modelo de classificação:",
        options=["Clássico", "Quântico"]
    )
col1, col2, col3 = st.columns(3)



# Coluna 1
with col1:
    
    valor = st.number_input("Valor da transação (em R$):", min_value=0.0, value=100.0, format="%.2f")
    fim_de_semana = st.selectbox(
        "A transação foi no final de semana?", 
        options=[0, 1], 
        format_func=lambda x: "Sim" if x == 1 else "Não"
    )

# Coluna 2
with col2:    
    numero_transacoes = st.number_input("Número de transações no dia:", min_value=1, value=5)
    perfil = st.selectbox("Perfil do cliente:", options=["Aposentado", "Estudante", "Não Aposentado"])


# Coluna 3
with col3:
    tipo_transacao = st.selectbox("Tipo de transação:", options=["Compra", "Saque", "Transferência"])
    cidade = st.selectbox("Cidade da transação:", options=[
        "São Paulo", "Rio de Janeiro", "Belo Horizonte", "Brasília", "Salvador", 
        "Fortaleza", "Curitiba", "Manaus", "Recife", "Porto Alegre", 
        "Vitória", "Goiânia", "Belém", "São Luís", "Maceió", 
        "Natal", "Campo Grande", "João Pessoa", "Aracaju", "Teresina", 
        "Cuiabá", "Macapá", "Rio Branco", "Boa Vista", "Palmas", 
        "Florianópolis", "Porto Velho"
    ])
col4, col5 = st.columns(2)
with col4:
    idade_cliente = st.slider("Idade do cliente:", min_value=18, max_value=80, value=30)
with col5:
    tempo = st.slider(
            "Hora da transação (0-24h):",
            min_value=0.0,  # Início do intervalo
            max_value=24.0, # Fim do intervalo
            value=12.0,     # Valor inicial
            step=0.5        # Incremento de 30 minutos
        )

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
            st.info("Acurácia do modelo clássico: 97%")  # Atualize com a acurácia real
        else:
            st.info("Acurácia do modelo quântico: 93%")  # Atualize com a acurácia real

    except Exception as e:
        st.error(f"Erro na classificação: {e}")

        # st.write("---")
st.markdown(
    """
    <div style='margin:0; padding:0px; text-align: center; color: gray;'>
        Desenvolvido por Weslley Rosa Prado<br>
        Orientador: Professor Doutor José Alexandre Nogueira<br>
        Projeto Prático Baseado no Trabalho de Conclusão de Curso de Física<br>
        Universidade Federal do Espírito Santo - UFES
    </div>
    """,
    unsafe_allow_html=True
)