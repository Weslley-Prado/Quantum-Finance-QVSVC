quantum_guard_backend/
│
├── app/
│   └── __init__.py
│   └── api.py              # Endpoints Flask
│
├── use_cases/
│   └── __init__.py
│   └── classification_of_transaction.py  # Lógica de classificação da transação
│
├── domain/
│   └── __init__.py
│   └── rules_transaction_models.py          # Definindo os modelos e lógica
│
├── adapters/
│   └── __init__.py
│   └── connection_models.py   # Código para carregar modelos .pkl
│
├── infra
│   └── model_of_classical_machine_learning
│       └── modelo_anti_fraude.pkl
│   └── model_of_quantum_machine_learning
│       └── modelo_anti_fraude_quantum.pkl
│
└── requirements.txt