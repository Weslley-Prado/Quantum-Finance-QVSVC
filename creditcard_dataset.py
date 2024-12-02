import pandas as pd
import numpy as np

# Semente para reprodutibilidade
np.random.seed(42)

# Configuração do tamanho do dataset
n_samples = 5000

# Gerar dados simulados
data = pd.DataFrame({
    "id_transacao": np.arange(1, n_samples + 1),
    "valor": np.random.exponential(scale=200, size=n_samples).clip(0, 10000),  # Valores de 0 a 10.000 reais
    "tempo": np.random.uniform(0, 24, size=n_samples),  # Tempo da transação (horário do dia)
    "fim_de_semana": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),  # Fim de semana ou não
    "idade_cliente": np.random.randint(18, 80, size=n_samples),  # Idade dos clientes
    "tipo_transacao": np.random.choice(["compra", "saque", "transferencia"], size=n_samples, p=[0.7, 0.2, 0.1]),
    "pais": np.random.choice(["Brasil", "EUA", "Alemanha", "Japão"], size=n_samples, p=[0.7, 0.1, 0.1, 0.1]),
    "latitude": np.random.uniform(-30.0, -15.0, size=n_samples),  # Coordenadas simuladas para o Brasil
    "longitude": np.random.uniform(-60.0, -35.0, size=n_samples),
    "fraude": np.random.choice([0, 1], size=n_samples, p=[0.97, 0.03])  # 3% de fraudes
})

# Normalizar colunas de valor e tempo
data["valor"] = data["valor"].round(2)  # Arredondar valores para duas casas decimais
data["tempo"] = data["tempo"].round(2)

# Salvar o dataset em um arquivo CSV
data.to_csv("creditcard_dataset.csv", index=False)

print("Dataset criado e salvo como 'creditcard_dataset.csv'")