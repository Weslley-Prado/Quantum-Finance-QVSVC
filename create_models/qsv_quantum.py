# Importando bibliotecas necessárias
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from qiskit.circuit.library import ZZFeatureMap
from qiskit_aer import Aer
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib  # Para salvar o modelo treinado
import matplotlib.pyplot as plt
import seaborn as sns

# Passo 1: Carregar os dados
data = pd.read_csv("../creditcard_dataset_com_cidades_e_regras.csv")
df = pd.DataFrame(data)

# Exibindo a distribuição de fraudes no dataset
print(df['fraude'].value_counts())

# Criar a pasta para armazenar os gráficos e documentação
output_dir = 'documentacao_modelo_quantico'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Gerar gráfico de dispersão da distribuição de fraudes
plt.figure(figsize=(6, 4))
sns.countplot(x='fraude', data=df)
plt.title('Distribuição de Fraudes')
plt.xlabel('Fraude (0 = Não, 1 = Sim)')
plt.ylabel('Contagem')
plt.savefig(os.path.join(output_dir, 'distribuicao_fraudes.png'))
plt.close()

# Pré-processamento
X = df.drop(columns=['fraude', 'id_transacao'])
y = df['fraude']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definindo as transformações para as colunas numéricas e categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), ['valor', 'tempo', 'idade_cliente', 'numero_transacoes']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['tipo_transacao', 'cidade', 'perfil', 'fim_de_semana'])
    ])

# Definindo a redução de dimensionalidade com PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=8)  # Número de componentes principais

# Configurando o QSVC com o kernel quântico
feature_map = ZZFeatureMap(feature_dimension=8, reps=1, entanglement='linear')
simulator = Aer.get_backend('aer_simulator')
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

model = QSVC(quantum_kernel=quantum_kernel)

# Criando o pipeline de treinamento e classificação
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', pca),  # Redução de dimensionalidade
    ('classifier', model)  # QSVC
])

# Treinando com um subconjunto (para prototipação)
sample_size = 400  # Usando um subconjunto inicial para acelerar o protótipo
X_train_sample = X_train[:sample_size]
y_train_sample = y_train.iloc[:sample_size].reset_index(drop=True)

# Treinando o modelo
pipeline.fit(X_train_sample, y_train_sample)

# Salvando o modelo treinado
joblib.dump(pipeline, os.path.join(output_dir, 'modelo_anti_fraude_quantum.pkl'))

# Avaliação do modelo
y_pred = pipeline.predict(X_test)  # O pipeline já cuida de transformações e PCA

# Relatório de classificação
report = classification_report(y_test, y_pred)
with open(os.path.join(output_dir, 'relatorio_classificacao_quantum.txt'), 'w') as f:
    f.write(report)

# Gerar a matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Fraude', 'Fraude'], yticklabels=['Não Fraude', 'Fraude'])
plt.title('Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.savefig(os.path.join(output_dir, 'matriz_confusao_quantum.png'))
plt.close()

# Documentação sobre colunas usadas
numerical_columns = preprocessor.transformers_[0][2]
categorical_columns = preprocessor.transformers_[1][2]
encoder = preprocessor.transformers_[1][1]  # OneHotEncoder
onehot_columns = encoder.get_feature_names_out(categorical_columns)

all_columns = numerical_columns + list(onehot_columns)

# Salvando as colunas utilizadas para o treinamento
with open(os.path.join(output_dir, 'colunas_usadas_quantum.txt'), 'w') as f:
    f.write("Colunas numéricas:\n")
    f.write("\n".join(numerical_columns) + "\n\n")
    f.write("Colunas categóricas:\n")
    f.write("\n".join(categorical_columns) + "\n\n")
    f.write("Colunas One-Hot:\n")
    f.write("\n".join(onehot_columns) + "\n\n")
    f.write("Todas as colunas utilizadas:\n")
    f.write("\n".join(all_columns) + "\n")

# Confirmação de que o modelo foi salvo com sucesso
print("Modelo treinado e salvo com sucesso em 'modelo_anti_fraude_quantum.pkl'")

# Confirmação de que a documentação foi gerada com sucesso
print("Documentação gerada com sucesso na pasta 'documentacao_modelo_quantico'.")
