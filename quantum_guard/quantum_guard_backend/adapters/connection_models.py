import joblib

class ModeloAdapter:
    def __init__(self, caminho_modelo_classico, caminho_modelo_quantico):
        """
        Carrega os modelos clássico e quântico a partir dos arquivos fornecidos.
        """
        # Carregar os modelos usando joblib
        self.modelo_classico = joblib.load(caminho_modelo_classico)
        self.modelo_quantico = joblib.load(caminho_modelo_quantico)

    def classificar(self, modelo_selecionado, dados):
        """
        Classifica os dados usando o modelo selecionado.

        :param modelo_selecionado: 'Clássico' ou 'Quantico' (modelo escolhido pelo usuário)
        :param dados: Dados para classificação (tipicamente um array ou DataFrame)
        :return: Previsão do modelo escolhido
        """

        # Selecionar o modelo com base na escolha
        if modelo_selecionado == "Clássico":
            modelo = self.modelo_classico
        else:
            modelo = self.modelo_quantico

        # Realizar a classificação e retornar o resultado
        return modelo.predict(dados)  # Geralmente, dados é uma lista ou array
