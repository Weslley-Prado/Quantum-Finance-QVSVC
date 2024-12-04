from flask import Flask, request, jsonify
from flask_cors import CORS
from adapters.connection_models import ModeloAdapter
import os
import pandas as pd

app = Flask(__name__)
CORS(app)  # Permite CORS para todas as origens




# Determina o caminho do arquivo com base no diretório do script atual
base_dir = os.path.dirname(os.path.abspath(__file__))

# Caminhos dos modelos .pkl
caminho_modelo_classico = os.path.join(base_dir, 'infra', 'model_of_classical_machine_learning', 'modelo_anti_fraude.pkl')
caminho_modelo_quantico = os.path.join(base_dir, 'infra', 'model_of_quantum_machine_learning', 'modelo_anti_fraude_quantum.pkl')

# Criar adaptador para carregar os modelos
modelo_adapter = ModeloAdapter(caminho_modelo_classico, caminho_modelo_quantico)

@app.route('/classificar', methods=['POST'])
def classificar():
    try:
        # Receber os dados no formato JSON
        dados = request.get_json()

        # Campos obrigatórios
        campos_obrigatorios = [
            'valor', 'tempo', 'fim_de_semana', 'idade_cliente',
            'numero_transacoes', 'tipo_transacao', 'cidade', 'perfil'
        ]

        # Verificar se todos os campos obrigatórios estão presentes
        campos_faltando = [campo for campo in campos_obrigatorios if campo not in dados]
        if campos_faltando:
            return jsonify({"erro": f"Campos faltando: {', '.join(campos_faltando)}"}), 400

        # Validar os tipos de dados (como Streamlit)
        try:
            novos_dados = pd.DataFrame([{
                "valor": dados["valor"],
                "tempo": dados["tempo"],
                "fim_de_semana": dados["fim_de_semana"],
                "idade_cliente": dados["idade_cliente"],
                "numero_transacoes": dados["numero_transacoes"],
                "tipo_transacao": dados["tipo_transacao"],
                "cidade": dados["cidade"],
                "perfil": dados["perfil"]}])
                            
            # entrada_modelo = novos_dados.to_numpy()
            print(novos_dados)
        except ValueError as e:
            return jsonify({"erro": f"Erro na validação dos dados: {str(e)}"}), 400

        # Modelo selecionado
        modelo_selecionado = dados.get('modelo_selecionado', 'Clássico')

        # Realizar a classificação
        resultado = modelo_adapter.classificar(modelo_selecionado, novos_dados)

        # Resposta do modelo
        resposta = "Alto Risco de Fraude" if resultado[0] == 1 else "Baixo ou Sem Risco de Fraude"
        return jsonify({"resultado": resposta}), 200

    except Exception as e:
        return jsonify({"erro": f"Erro no processamento: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
