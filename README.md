[Documentação em Português](#quantum-finance-qvsvc--português)
[Documentation in English](#quantum-finance-qvsvc--english)

---

# Quantum Finance QVSVC 💰 (Português)

Análise de Transações Financeiras com Machine Learning Clássico e Quântico

---

## Descrição do Projeto

O **Quantum Finance QVSVC** é uma aplicação projetada para classificar transações financeiras simuladas como **Fraudulentas** ou **Não Fraudulentas** usando duas abordagens distintas:
- **SVM Clássico (Support Vector Machine)**: Um modelo tradicional de aprendizado de máquina com kernel RBF.
- **QSVM (Quantum SVM)**: Um modelo de aprendizado de máquina quântico que utiliza um kernel quântico baseado na mecânica quântica.

Esta aplicação compara as eficácias de um modelo clássico e um modelo quântico para classificação de dados, demonstrando as vantagens e limitações da computação quântica aplicada à detecção de fraudes.

## Arquitetura do Projeto

### 1. **Geração de Dados Simulados**

Os dados de transações financeiras são simulados usando valores aleatórios, onde cada transação é representada por:
- **Valor da Transação**: Representa um montante normalizado entre 0 e 1.
- **Tempo da Transação**: Representa um intervalo de tempo normalizado.

Cada transação é rotulada como:
- **0 (Não Fraudulenta)** ou **1 (Fraudulenta)**

### 2. **Divisão de Dados**

Os dados são divididos em dois conjuntos:
- **Treinamento**: 70% dos dados para treinar os modelos.
- **Teste**: 30% dos dados para avaliar a performance dos modelos.

### 3. **Modelos de Classificação**

A aplicação utiliza dois modelos distintos para a classificação das transações:

- **SVM Clássico**: Um modelo tradicional de SVM usando um kernel RBF, que é ótimo para dados com características não-lineares.
- **QSVM (Quantum SVM)**: Utiliza um kernel quântico configurado com o `ZZFeatureMap` para capturar relações complexas entre as transações. Este modelo usa o backend do Qiskit Aer, um simulador quântico.

### 4. **Interface do Usuário com Streamlit**

A interface, desenvolvida em **Streamlit**, permite que o usuário insira manualmente os valores de uma transação para a classificação, fornecendo um entendimento prático do uso dos modelos:

- **Valor da Transação**: Entrada de um valor entre 0 e 1.
- **Tempo da Transação**: Entrada de um intervalo de tempo entre 0 e 1.

O usuário pode escolher qual modelo (SVM Clássico ou QSVM) quer usar para a classificação da transação, e a interface exibe o resultado da classificação, indicando se a transação é **Fraudulenta** ou **Não Fraudulenta**.

### 5. **Avaliação dos Modelos**

Após as previsões, a aplicação calcula a acurácia dos dois modelos usando os dados de teste e exibe as métricas de desempenho para comparação direta.

---

## Uso da Aplicação

1. **Insira** o valor e o tempo da transação nos campos apropriados.
2. **Selecione o modelo** desejado para realizar a classificação (SVM Clássico ou QSVM).
3. **Veja o resultado**, indicando se a transação é classificada como Fraudulenta ou Não Fraudulenta.
4. **Compare as acurácias** dos modelos na seção de métricas para avaliar o desempenho de cada abordagem.

---

## Desenvolvimento e Orientação

Desenvolvido por **Weslley Rosa Prado**  
Orientador: **Professor Doutor José Alexandre Nogueira**  
Projeto baseado no Trabalho de Conclusão de Curso de Física  
**Universidade Federal do Espírito Santo - UFES**

---

Este projeto é uma demonstração prática da aplicação da computação quântica na área de aprendizado de máquina, explorando potenciais melhorias na detecção de padrões complexos, como em dados financeiros.

---

# Quantum Finance QVSVC 💰 (English)

Analysis of Financial Transactions with Classical and Quantum Machine Learning

---

## Project Description

**Quantum Finance QVSVC** is an application designed to classify simulated financial transactions as either **Fraudulent** or **Non-Fraudulent** using two distinct approaches:
- **Classical SVM (Support Vector Machine)**: A traditional machine learning model with an RBF kernel.
- **QSVM (Quantum SVM)**: A quantum machine learning model that uses a quantum kernel based on quantum mechanics.

This application compares the performance of a classical model and a quantum model for data classification, demonstrating the advantages and limitations of quantum computing applied to fraud detection.

## Project Architecture

### 1. **Generating Simulated Data**

The financial transaction data is simulated using random values, where each transaction is represented by:
- **Transaction Amount**: Represents a normalized amount between 0 and 1.
- **Transaction Time**: Represents a normalized time interval.

Each transaction is labeled as:
- **0 (Non-Fraudulent)** or **1 (Fraudulent)**

### 2. **Data Splitting**

The data is split into two sets:
- **Training**: 70% of the data is used to train the models.
- **Testing**: 30% of the data is used to evaluate the models’ performance.

### 3. **Classification Models**

The application uses two distinct models for transaction classification:

- **Classical SVM**: A traditional SVM model using an RBF kernel, which is excellent for data with non-linear characteristics.
- **QSVM (Quantum SVM)**: Utilizes a quantum kernel configured with the `ZZFeatureMap` to capture complex relationships between transactions. This model uses Qiskit Aer as a quantum backend simulator.

### 4. **User Interface with Streamlit**

The interface, built with **Streamlit**, allows the user to manually input transaction values for classification, providing a practical understanding of model usage:

- **Transaction Amount**: Input a value between 0 and 1.
- **Transaction Time**: Input a time interval between 0 and 1.

The user can choose which model (Classical SVM or QSVM) to use for transaction classification, and the interface displays the classification result, indicating whether the transaction is **Fraudulent** or **Non-Fraudulent**.

### 5. **Model Evaluation**

After predictions, the application calculates the accuracy of both models using the test data and displays performance metrics for a direct comparison.

---

## Application Usage

1. **Enter** the transaction amount and time in the respective fields.
2. **Select the model** you want to use for classification (Classical SVM or QSVM).
3. **View the result**, indicating if the transaction is classified as Fraudulent or Non-Fraudulent.
4. **Compare model accuracies** in the metrics section to evaluate the performance of each approach.

---

## Development and Supervision

Developed by **Weslley Rosa Prado**  
Supervisor: **Professor Dr. José Alexandre Nogueira**  
Project based on the Physics Undergraduate Thesis  
**Federal University of Espírito Santo - UFES**

---

This project is a practical demonstration of the application of quantum computing in the field of machine learning, exploring potential improvements in detecting complex patterns, such as in financial data.