class Transacao:
    def __init__(self, valor, tempo, fim_de_semana, idade_cliente, numero_transacoes, tipo_transacao, cidade, perfil):
        self.valor = valor
        self.tempo = tempo
        self.fim_de_semana = fim_de_semana
        self.idade_cliente = idade_cliente
        self.numero_transacoes = numero_transacoes
        self.tipo_transacao = tipo_transacao
        self.cidade = cidade
        self.perfil = perfil

    def to_dict(self):
        return {
            "valor": self.valor,
            "tempo": self.tempo,
            "fim_de_semana": self.fim_de_semana,
            "idade_cliente": self.idade_cliente,
            "numero_transacoes": self.numero_transacoes,
            "tipo_transacao": self.tipo_transacao,
            "cidade": self.cidade,
            "perfil": self.perfil
        }
