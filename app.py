import streamlit as st
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from qiskit_aer import Aer
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

# Configuração do estilo da página
st.set_page_config(page_title="Quantum Finance QVSVC", page_icon="💰", layout="centered")

# Cabeçalho
st.markdown("<h1 style='text-align: center; color: #4B0082;'>Quantum Finance QVSVC</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #6A5ACD;'>Análise de Transações Financeiras com Machine Learning Clássico e Quântico</h4>", unsafe_allow_html=True)
st.write("---")

# Dados simulados de transações financeiras
np.random.seed(42)
n_samples = 200
X = np.random.rand(n_samples, 2)  # Características: valor e tempo
y = np.random.choice([0, 1], size=n_samples)  # Classe: 0 (não fraudulenta), 1 (fraudulenta)

# Divisão dos dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinamento do SVM Clássico
svc_classical = SVC(kernel='rbf', gamma='scale')
svc_classical.fit(X_train, y_train)

# Configuração do Kernel Quântico para QSVM
backend = Aer.get_backend('aer_simulator')
feature_map = ZZFeatureMap(feature_dimension=2, reps=2, entanglement='linear')
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
qsvc = QSVC(quantum_kernel=quantum_kernel)
qsvc.fit(X_train, y_train)

# Interface com o Streamlit
st.subheader("Insira as características da transação para classificação")

# Entrada de dados
valor = st.number_input("Valor da Transação - normalizado entre 0 e 1, onde 1 seria um valor alto", min_value=0.0, max_value=1.0, value=0.5)
tempo = st.number_input("Tempo da Transação - normalizado, representando o intervalo de tempo em que a transação aconteceu", min_value=0.0, max_value=1.0, value=0.5)
entrada = np.array([[valor, tempo]])

# Previsão com o SVM Clássico
if st.button("Classificar com SVM Clássico"):
    pred_classico = svc_classical.predict(entrada)
    resultado_classico = "Fraudulenta" if pred_classico[0] == 1 else "Não Fraudulenta"
    st.success(f"Resultado com SVM Clássico: **{resultado_classico}**")

# Previsão com o QSVM
if st.button("Classificar com QSVM"):
    pred_quantico = qsvc.predict(entrada)
    resultado_quantico = "Fraudulenta" if pred_quantico[0] == 1 else "Não Fraudulenta"
    st.success(f"Resultado com QSVM: **{resultado_quantico}**")

# Comparação de acurácia
y_pred_classical = svc_classical.predict(X_test)
accuracy_classical = accuracy_score(y_test, y_pred_classical)

y_pred_quantum = qsvc.predict(X_test)
accuracy_quantum = accuracy_score(y_test, y_pred_quantum)

st.write("---")
st.write("### Acurácia dos Modelos:")
st.write(f"- Acurácia SVM Clássico: **{accuracy_classical:.4f}**")
st.write(f"- Acurácia QSVM: **{accuracy_quantum:.4f}**")

# Rodapé
st.write("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Desenvolvido por Weslley Rosa Prado<br>
        Orientador: Professor Doutor José Alexandre Nogueira<br>
        Projeto Prático Baseado no Trabalho de Conclusão de Curso de Física<br>
        Universidade Federal do Espírito Santo - UFES
    </div>
    """,
    unsafe_allow_html=True
)
# 

