# import pandas as pd
# import numpy as np

# # Semente para reprodutibilidade
# np.random.seed(42)

# # Configuração do tamanho do dataset
# n_samples = 20000

# # Lista das 27 capitais brasileiras
# capitais_brasileiras = [
#     "São Paulo", "Rio de Janeiro", "Belo Horizonte", "Brasília", "Salvador", 
#     "Fortaleza", "Curitiba", "Manaus", "Recife", "Porto Alegre", 
#     "Vitória", "Goiânia", "Belém", "São Luís", "Maceió", "Natal", 
#     "Campo Grande", "João Pessoa", "Aracaju", "Teresina", "Cuiabá", 
#     "Macapá", "Rio Branco", "Boa Vista", "Palmas", "Florianópolis", "Porto Velho"
# ]

# # Função para categorizar perfis de acordo com a idade
# def categorizar_perfil(idade):
#     if idade >= 60:
#         return "Aposentado"
#     elif idade >= 18 and idade <= 25:
#         return "Estudante"
#     elif idade > 25 and idade <= 45:
#         return np.random.choice(["Engenheiro", "Bancário", "Investidor", "Comerciante"])
#     elif idade > 45 and idade <= 60:
#         return np.random.choice(["Bancário", "Investidor", "Comerciante"])

# # Gerar dados simulados
# data = pd.DataFrame({
#     "id_transacao": np.arange(1, n_samples + 1),
#     "valor": np.random.exponential(scale=200, size=n_samples).clip(1000, 100000),  # Valores simulados entre 0 e 10.000 reais
#     "tempo": np.random.uniform(0, 24, size=n_samples),  # Tempo da transação (horário do dia)
#     "fim_de_semana": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),  # Indica fim de semana (1) ou não (0)
#     "idade_cliente": np.random.randint(18, 80, size=n_samples),  # Idades simuladas entre 18 e 80 anos
#     "tipo_transacao": np.random.choice(["compra", "saque", "transferencia"], size=n_samples, p=[0.7, 0.2, 0.1]),
#     "cidade": np.random.choice(capitais_brasileiras, size=n_samples),  # Aleatoriza as cidades
#     "fraude": np.random.choice([0, 1], size=n_samples, p=[0.95, 0.35])  # 95% não fraude, 5% fraude
# })

# # Categorizar perfil com base na idade
# data["perfil"] = data["idade_cliente"].apply(categorizar_perfil)

# # Modificar o número de transações de acordo com o perfil
# data["numero_transacoes"] = data["perfil"].apply(lambda perfil: np.random.randint(1, 6) if perfil in ["Bancário", "Investidor", "Comerciante"] else np.random.randint(1, 2))

# # Implementar as regras de fraude
# valor_transacao = data["valor"] * data["numero_transacoes"]

# # Regra 1: Transações até 350 reais, não bancário/investidor/comerciante, 1-2 horas entre transações
# mask1 = (
#     (valor_transacao >= 3000.00) & 
#     (~data["perfil"].isin(["Bancário", "Investidor", "Comerciante"])) & 
#     ((data["tempo"] % 2) < 2)
# )
# data.loc[mask1, "fraude"] = 1

# # Regra 2: Compras ou saques acima de 20 mil no fim de semana
# mask2 = (data["valor"] > 20000) & (data["fim_de_semana"] == 1) & (data["tipo_transacao"].isin(["compra", "saque"]))
# data.loc[mask2, "fraude"] = 1

# # Regra 3: Aposentados com transações acima de 5 mil mais de uma vez por dia
# mask3 = (data["perfil"] == "Aposentado") & (data["valor"] > 5000) & (data["numero_transacoes"] > 1)
# data.loc[mask3, "fraude"] = 1

# # Regra 4: Transações acima de 100 mil mais de uma vez por dia
# mask4 = (data["valor"] > 100000) & (data["numero_transacoes"] > 1)
# data.loc[mask4, "fraude"] = 1

# # Normalizar colunas de valor e tempo
# data["valor"] = data["valor"].round(2)  # Arredondar valores para duas casas decimais
# data["tempo"] = data["tempo"].round(2)

# # Salvar o dataset em um arquivo CSV
# data.to_csv("creditcard_dataset_com_regras.csv", index=False)

# print("Dataset criado e salvo como 'creditcard_dataset_com_regras.csv'")


#  Regras adicionais para cidades e estelionato

# # Regra 1: Cidades com Alto Índice de Estelionato
# alto_risco = ["São Paulo", "Rio de Janeiro", "Salvador"]
# mask1 = (data["cidade"].isin(alto_risco)) & (data["valor"] > 10000)
# data.loc[mask1, "fraude"] = 1

# # Regra 2: Frequência de Transações em Cidades Grandes
# capitais_grandes = ["São Paulo", "Rio de Janeiro", "Brasília"]
# mask2 = (data["cidade"].isin(capitais_grandes)) & (data["numero_transacoes"] > 3)
# data.loc[mask2, "fraude"] = 1

# # Regra 3: Horário Não Comercial em Capitais
# horario_comercial = (data["tempo"] >= 8) & (data["tempo"] <= 22)
# mask3 = (~horario_comercial) & (data["cidade"].isin(capitais_grandes)) & (data["valor"] > 5000)
# data.loc[mask3, "fraude"] = 1

# # Regra 4: Cidades Pequenas com Transações Altas
# capitais_pequenas = ["Boa Vista", "Rio Branco", "Palmas"]
# mask4 = (data["cidade"].isin(capitais_pequenas)) & (data["valor"] > 50000)
# data.loc[mask4, "fraude"] = 1

# # Regra 5: Perfis e Localidades Incompatíveis
# mask5 = (data["perfil"] == "Estudante") & (data["cidade"] == "Brasília") & (data["valor"] > 3000)
# data.loc[mask5, "fraude"] = 1

# # Regra 6: Finais de Semana em Cidades Turísticas
# turisticas = ["Salvador", "Recife", "Fortaleza", "Rio de Janeiro"]
# mask6 = (data["cidade"].isin(turisticas)) & (data["fim_de_semana"] == 1) & (data["perfil"].isin(["Aposentado", "Estudante"])) & (data["valor"] > 2500)
# data.loc[mask6, "fraude"] = 1

# # Regra 7: Transações em Cidades Diferentes em Intervalo Curto
# # Simulação de mudança de cidades com IDs consecutivos (ilustrativo)
# # Caso houvesse identificação única por cliente e timestamp, implementar lógica temporal.
# # Aqui simulamos como exemplo:
# data["cidade_shift"] = data["cidade"].shift(1)
# data["tempo_shift"] = data["tempo"].shift(1)
# mask7 = (data["cidade"] != data["cidade_shift"]) & ((data["tempo"] - data["tempo_shift"]).abs() < 1)
# data.loc[mask7, "fraude"] = 1

# # Remover colunas temporárias usadas no processo
# data.drop(columns=["cidade_shift", "tempo_shift"], inplace=True)

# # Salvar dataset modificado
# data.to_csv("creditcard_dataset_com_cidades_e_regras.csv", index=False)

# print("Regras adicionais aplicadas. Dataset salvo como 'creditcard_dataset_com_cidades_e_regras.csv'")

import numpy as np
import pandas as pd

# Função para categorizar perfil com base na idade
def categorizar_perfil(idade):
    if idade < 30:
        return "Estudante"
    elif 30 <= idade < 60:
        return "Não Aposentado"
    else:
        return "Aposentado"

# Lista das capitais brasileiras
capitais_brasileiras = [
    "São Paulo", "Rio de Janeiro", "Belo Horizonte", "Brasília", "Salvador", 
    "Fortaleza", "Curitiba", "Manaus", "Recife", "Porto Alegre", 
    "Vitória", "Goiânia", "Belém", "São Luís", "Maceió", "Natal", 
    "Campo Grande", "João Pessoa", "Aracaju", "Teresina", "Cuiabá", 
    "Macapá", "Rio Branco", "Boa Vista", "Palmas", "Florianópolis", "Porto Velho"
]

n_samples = 401  # Número total de amostras

# Gerar dados simulados
data = pd.DataFrame({
    "id_transacao": np.arange(1, n_samples + 1),
    "valor": np.random.exponential(scale=200, size=n_samples).clip(1000, 100000),  # Valores simulados entre 1000 e 100000 reais
    "tempo": np.random.uniform(0, 24, size=n_samples),  # Tempo da transação (horário do dia)
    "fim_de_semana": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),  # Indica fim de semana (1) ou não (0)
    "idade_cliente": np.random.randint(18, 80, size=n_samples),  # Idades simuladas entre 18 e 80 anos
    "tipo_transacao": np.random.choice(["compra", "saque", "transferencia"], size=n_samples, p=[0.7, 0.2, 0.1]),
    "cidade": np.random.choice(capitais_brasileiras, size=n_samples),  # Aleatoriza as cidades
})

# Categorizar perfil com base na idade
data["perfil"] = data["idade_cliente"].apply(categorizar_perfil)

# Modificar o número de transações de acordo com o perfil
data["numero_transacoes"] = data["perfil"].apply(lambda perfil: np.random.randint(1, 6) if perfil in ["Não Aposentado", "Estudante"] else np.random.randint(1,5))

# Inicialmente, todas as transações são não fraudulentas
data["fraude"] = 0

# Implementar as regras de fraude
valor_transacao = data["valor"] * data["numero_transacoes"]

# Regra 1: Transações até 350 reais, não bancário/investidor/comerciante, 1-2 horas entre transações
mask1 = (
    (valor_transacao >= 3000.00) & 
    ((data["tempo"] % 2) < 2)
)
data.loc[mask1, "fraude"] = 1

# Regra 3: Aposentados com transações acima de 5 mil mais de uma vez por dia
mask3 = (data["perfil"] == "Aposentado") & (data["valor"] > 5000) & (data["numero_transacoes"] > 1)
data.loc[mask3, "fraude"] = 1

# Regra 4: Transações acima de 100 mil mais de uma vez por dia
mask4 = (data["valor"] > 100000) & (data["numero_transacoes"] > 1)
data.loc[mask4, "fraude"] = 1

# Regras adicionais para cidades e estelionato

# Regra 1: Cidades com Alto Índice de Estelionato
alto_risco = ["São Paulo", "Rio de Janeiro", "Salvador"]
mask5 = (data["cidade"].isin(alto_risco)) & (data["valor"] > 10000)
data.loc[mask5, "fraude"] = 1

# Regra 2: Frequência de Transações em Cidades Grandes
capitais_grandes = ["São Paulo", "Rio de Janeiro", "Brasília"]
mask6 = (data["cidade"].isin(capitais_grandes)) & (data["numero_transacoes"] > 3)
data.loc[mask6, "fraude"] = 1

# Regra 3: Horário Não Comercial em Capitais
horario_comercial = (data["tempo"] >= 8) & (data["tempo"] <= 22)
mask7 = (~horario_comercial) & (data["cidade"].isin(capitais_grandes)) & (data["valor"] > 5000)
data.loc[mask7, "fraude"] = 1

# Regra 4: Cidades Pequenas com Transações Altas
capitais_pequenas = ["Boa Vista", "Rio Branco", "Palmas"]
mask8 = (data["cidade"].isin(capitais_pequenas)) & (data["valor"] > 50000)
data.loc[mask8, "fraude"] = 1

# Regra 5: Perfis e Localidades Incompatíveis
mask9 = (data["perfil"] == "Estudante") & (data["cidade"] == "Brasília") & (data["valor"] > 3000)
data.loc[mask9, "fraude"] = 1

# Regra 6: Finais de Semana em Cidades Turísticas
turisticas = ["Salvador", "Recife", "Fortaleza", "Rio de Janeiro"]
mask10 = (data["cidade"].isin(turisticas)) & (data["fim_de_semana"] == 1) & (data["perfil"].isin(["Aposentado", "Estudante"])) & (data["valor"] > 2500)
data.loc[mask10, "fraude"] = 1

# Regra 7: Transações em Cidades Diferentes em Intervalo Curto
# Simulação de mudança de cidades com IDs consecutivos (ilustrativo)
data["cidade_shift"] = data["cidade"].shift(1)
data["tempo_shift"] = data["tempo"].shift(1)
mask11 = (data["cidade"] != data["cidade_shift"]) & ((data["tempo"] - data["tempo_shift"]).abs() < 1)
data.loc[mask11, "fraude"] = 1

# Remover colunas temporárias usadas no processo
data.drop(columns=["cidade_shift", "tempo_shift"], inplace=True)

# Salvar dataset modificado
data.to_csv("creditcard_dataset_com_cidades_e_regras_classical.csv", index=False)

# Verificando a distribuição de fraudes
print(data["fraude"].value_counts())
