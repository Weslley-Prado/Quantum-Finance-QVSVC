# Importando as bibliotecas necessárias
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib  # Biblioteca para salvar o modelo treinado
import matplotlib.pyplot as plt
import seaborn as sns

# Passo 1: Carregar os dados
# Carregamos os dados de um arquivo CSV
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

# Dividindo os dados em treino e teste (70% para treino e 30% para teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definindo as transformações para as colunas numéricas e categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), ['valor', 'tempo', 'idade_cliente', 'numero_transacoes']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['tipo_transacao', 'cidade', 'perfil', 'fim_de_semana'])
    ])

# Passo 4: Criar o pipeline
# O pipeline engloba o pré-processamento e a classificação usando SVC (Máquina de Vetores de Suporte com kernel linear)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='linear', random_state=42))
])

# Treinando o modelo com os dados de treino
pipeline.fit(X_train, y_train)

# Salvando o modelo treinado em um arquivo usando joblib
joblib.dump(pipeline, os.path.join(output_dir, 'modelo_anti_fraude.pkl'))

# Passo 5: Avaliar o modelo
# Realizando previsões com os dados de teste
y_pred = pipeline.predict(X_test)

# Imprimir relatório de classificação (precisão, recall, f1-score) e salvar como texto
report = classification_report(y_test, y_pred)
with open(os.path.join(output_dir, 'relatorio_classificacao.txt'), 'w') as f:
    f.write(report)

# Gerar a matriz de confusão para avaliar os resultados do modelo
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Fraude', 'Fraude'], yticklabels=['Não Fraude', 'Fraude'])
plt.title('Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Real')
# Salvando a matriz de confusão como imagem
plt.savefig(os.path.join(output_dir, 'matriz_confusao.png'))
plt.close()

# Confirmação de que o modelo foi salvo com sucesso
print("Modelo treinado e salvo com sucesso em 'modelo_anti_fraude.pkl'")

# Gerar e salvar a lista de colunas utilizadas para o treinamento
# Colunas numéricas que foram escalonadas
numerical_columns = preprocessor.transformers_[0][2]
# Colunas categóricas que foram transformadas com One-Hot Encoding
categorical_columns = preprocessor.transformers_[1][2]
encoder = preprocessor.transformers_[1][1]  # OneHotEncoder
onehot_columns = encoder.get_feature_names_out(categorical_columns)

# Todas as colunas utilizadas no treinamento (numéricas + One-Hot)
all_columns = numerical_columns + list(onehot_columns)

# Salvando as colunas utilizadas para o treinamento em arquivo de texto
with open(os.path.join(output_dir, 'colunas_usadas.txt'), 'w') as f:
    f.write("Colunas numéricas:\n")
    f.write("\n".join(numerical_columns) + "\n\n")
    f.write("Colunas categóricas:\n")
    f.write("\n".join(categorical_columns) + "\n\n")
    f.write("Colunas One-Hot:\n")
    f.write("\n".join(onehot_columns) + "\n\n")
    f.write("Todas as colunas utilizadas:\n")
    f.write("\n".join(all_columns) + "\n")

# Confirmação de que a documentação foi gerada com sucesso
print("Documentação gerada com sucesso na pasta 'documentacao_modelo'.")
