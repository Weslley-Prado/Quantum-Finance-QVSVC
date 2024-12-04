from domain.rules_transaction_models import Transacao

class TransacaoClassificacao:
    def __init__(self, modelo_adapter):
        self.modelo_adapter = modelo_adapter

    def classificar_transacao(self, dados, modelo_selecionado):
        transacao = Transacao(**dados)
        dados_dict = transacao.to_dict()
        return self.modelo_adapter.classificar(modelo_selecionado, dados_dict)
