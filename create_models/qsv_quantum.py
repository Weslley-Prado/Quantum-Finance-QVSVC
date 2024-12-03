# Importando as bibliotecas necessárias
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from qiskit_aer import Aer
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Passo 1: Carregar os dados
# Carregar o dataset
data = pd.read_csv("../creditcard_dataset_com_cidades_e_regras.csv")

# Convertendo os dados para um DataFrame do pandas
df = pd.DataFrame(data)

# Exibindo a distribuição de fraudes no dataset
print(df['fraude'].value_counts())

# Passo 2: Criar a pasta para armazenar os gráficos e documentação
# Criamos uma pasta chamada 'documentacao_modelo' para salvar gráficos, tabelas e o modelo treinado
output_dir = 'documentacao_modelo'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Gerar gráfico de dispersão da distribuição de fraudes
plt.figure(figsize=(6, 4))
sns.countplot(x='fraude', data=df)
plt.title('Distribuição de Fraudes')
plt.xlabel('Fraude (0 = Não, 1 = Sim)')
plt.ylabel('Contagem')
# Salvando o gráfico na pasta de documentação
plt.savefig(os.path.join(output_dir, 'distribuicao_fraudes.png'))
plt.close()

# Gerar gráfico de dispersão entre valor da transação e idade do cliente
plt.figure(figsize=(6, 4))
sns.scatterplot(x='valor', y='idade_cliente', hue='fraude', data=df, palette='coolwarm')
plt.title('Dispersão de Valor vs Idade do Cliente')
plt.xlabel('Valor da Transação')
plt.ylabel('Idade do Cliente')
# Salvando o gráfico na pasta de documentação
plt.savefig(os.path.join(output_dir, 'dispersao_valor_idade.png'))
plt.close()

# Gerar tabela com amostra dos dados (primeiras linhas)
df_sample = df.head()
# Salvando a amostra dos dados como CSV
df_sample.to_csv(os.path.join(output_dir, 'amostra_dados.csv'), index=False)

# Passo 3: Pré-processamento
# Separando variáveis independentes (X) e dependentes (y)
X = df.drop(columns=['fraude', 'id_transacao'])
y = df['fraude']

# Balanceamento das classes com SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Dividindo os dados em treino e teste (70% para treino e 30% para teste)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Normalizando variáveis contínuas
scaler = MinMaxScaler()
X_train[['valor', 'tempo', 'idade_cliente']] = scaler.fit_transform(X_train[['valor', 'tempo', 'idade_cliente']])
X_test[['valor', 'tempo', 'idade_cliente']] = scaler.transform(X_test[['valor', 'tempo', 'idade_cliente']])

# Configuração do Kernel Quântico para QSVC
backend = Aer.get_backend('aer_simulator')
feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2, entanglement='linear')
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

# Passo 4: Documentação do circuito quântico
# Salvando o circuito quântico usado no Kernel
circuit = feature_map.construct_circuit([0]*X_train.shape[1])  # Circuito inicial para o número de features
circuit_image_path = os.path.join(output_dir, 'circuito_quantico.png')
circuit.draw(output='mpl')
plt.savefig(circuit_image_path)
plt.close()

# Documentação adicional sobre o circuito quântico
with open(os.path.join(output_dir, 'documentacao_circuito_quantico.txt'), 'w') as f:
    f.write("Descrição do Circuito Quântico:\n")
    f.write("O circuito quântico é baseado no ZZFeatureMap, com a entanglement 'linear' e 2 repetições.\n")
    f.write("Este circuito é utilizado para mapear as variáveis de entrada (features) em um espaço quântico.\n")
    f.write("O objetivo do kernel quântico é calcular a fidelidade entre os estados quânticos correspondentes\n")
    f.write("às amostras de treinamento para realizar a classificação.\n")
    f.write("\n")
    f.write(f"Imagem do circuito quântico salva em: {circuit_image_path}\n")
    f.write("Este circuito é um componente chave no algoritmo QSVC utilizado para classificação quântica.\n")

# Passo 5: Treinamento do QSVC
qsvc = QSVC(quantum_kernel=quantum_kernel)
qsvc.fit(X_train, y_train)

# Avaliar o modelo
y_pred = qsvc.predict(X_test)

# Métricas de desempenho
accuracy = qsvc.score(X_test, y_test)
print(f"Acurácia do QSVC: {accuracy:.4f}")
print("\nRelatório de Classificação:\n")
print(classification_report(y_test, y_pred, target_names=["Não Fraudulenta", "Fraudulenta"]))

# Passo 6: Salvar o modelo treinado
joblib.dump(qsvc, os.path.join(output_dir, 'quantum_model.pkl'))
print("Modelo QSVC salvo como 'quantum_model.pkl'")

# Gerar e salvar a lista de colunas utilizadas para o treinamento
numerical_columns = ['valor', 'tempo', 'idade_cliente', 'numero_transacoes']
categorical_columns = ['tipo_transacao', 'cidade', 'perfil', 'fim_de_semana']

# Salvando as colunas utilizadas para o treinamento em arquivo de texto
with open(os.path.join(output_dir, 'colunas_usadas.txt'), 'w') as f:
    f.write("Colunas numéricas:\n")
    f.write("\n".join(numerical_columns) + "\n\n")
    f.write("Colunas categóricas:\n")
    f.write("\n".join(categorical_columns) + "\n\n")

# Confirmação de que a documentação foi gerada com sucesso
print("Documentação gerada com sucesso na pasta 'documentacao_modelo'.")
