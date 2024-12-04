import { useState, ChangeEvent, FormEvent } from "react";

interface FormData {
  valor: number;
  tempo: number;
  fim_de_semana: number;
  idade_cliente: number;
  numero_transacoes: number;
  tipo_transacao: string;
  cidade: string;
  perfil: string;
  modelo_selecionado: string;
}

export default function Classificar() {
  const [formData, setFormData] = useState<FormData>({
    valor: 100.0,
    tempo: 12.0,
    fim_de_semana: 1,
    idade_cliente: 30,
    numero_transacoes: 5,
    tipo_transacao: "Compra",
    cidade: "São Paulo",
    perfil: "Aposentado",
    modelo_selecionado: "Clássico",
  });

  const [resultado, setResultado] = useState<string | null>(null);
  const [erro, setErro] = useState<string | null>(null);

  const handleChange = (
    e: ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    setFormData((prevState) => ({
      ...prevState,
      [name]: value,
    }));
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    try {
      const response = await fetch("http://localhost:5000/classificar", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (response.ok) {
        setResultado(data.resultado);
        setErro(null);
      } else {
        setErro(data.erro);
        setResultado(null);
      }
    } catch (err) {
      setErro("Erro ao se comunicar com o servidor.");
      setResultado(null);
    }
  };

  return (
    <div
      className="card p-5 shadow-lg mx-auto"
      style={{ width: "1200px", height: "700px" }}
    >
      <h1
        className="text-center text-primary mb-4"
        style={{ fontSize: "2.5rem", fontWeight: "bold" }}
      >
        Quantum Guard
      </h1>
      <p className="text-center mb-4" style={{ fontSize: "1.2rem" }}>
        Insira os dados da transação para classificação
      </p>

      <form onSubmit={handleSubmit}>
        <div className="row mb-1">
          <div className="col-md-4">
            <label className="form-label">Valor da Transação (R$):</label>
            <input
              type="number"
              name="valor"
              value={formData.valor}
              onChange={handleChange}
              className="form-control"
              style={{ borderRadius: "8px" }}
            />
          </div>
          <div className="col-md-4">
            <label className="form-label">Hora da Transação:</label>
            <input
              type="number"
              name="tempo"
              value={formData.tempo}
              onChange={handleChange}
              min={0}
              max={24}
              step={0.5}
              className="form-control"
              style={{ borderRadius: "8px" }}
            />
          </div>
          <div className="col-md-4">
            <label className="form-label">
              A transação ocorreu no final de semana?
            </label>
            <select
              name="fim_de_semana"
              value={formData.fim_de_semana}
              onChange={handleChange}
              className="form-select"
              style={{ borderRadius: "8px" }}
            >
              <option value={0}>Não</option>
              <option value={1}>Sim</option>
            </select>
          </div>
        </div>
        <div className="row mb-1">
          <div className="col-md-4">
            <label className="form-label">Idade do Cliente:</label>
            <input
              type="number"
              name="idade_cliente"
              value={formData.idade_cliente}
              onChange={handleChange}
              min={18}
              max={80}
              className="form-control"
              style={{ borderRadius: "8px" }}
            />
          </div>
          <div className="col-md-4">
            <label className="form-label">Número de Transações no Dia:</label>
            <input
              type="number"
              name="numero_transacoes"
              value={formData.numero_transacoes}
              onChange={handleChange}
              min={1}
              className="form-control"
              style={{ borderRadius: "8px" }}
            />
          </div>
          <div className="col-md-4">
            <label className="form-label">Tipo de Transação:</label>
            <select
              name="tipo_transacao"
              value={formData.tipo_transacao}
              onChange={handleChange}
              className="form-select"
              style={{ borderRadius: "8px" }}
            >
              <option value="Compra">Compra</option>
              <option value="Saque">Saque</option>
              <option value="Transferência">Transferência</option>
            </select>
          </div>
        </div>
        <div className="row mb-1">
          <div className="col-md-4">
            <label className="form-label">Cidade da Transação:</label>
            <select
              name="cidade"
              value={formData.cidade}
              onChange={handleChange}
              className="form-select"
              style={{ borderRadius: "8px" }}
            >
              <option value="São Paulo">São Paulo</option>
              <option value="Rio de Janeiro">Rio de Janeiro</option>
              <option value="Belo Horizonte">Belo Horizonte</option>
              <option value="Brasília">Brasília</option>
              <option value="Salvador">Salvador</option>
              <option value="Fortaleza">Fortaleza</option>
              <option value="Curitiba">Curitiba</option>
              <option value="Manaus">Manaus</option>
              <option value="Recife">Recife</option>
              <option value="Porto Alegre">Porto Alegre</option>
              <option value="Vitória">Vitória</option>
              <option value="Goiânia">Goiânia</option>
              <option value="Belém">Belém</option>
              <option value="São Luís">São Luís</option>
              <option value="Maceió">Maceió</option>
              <option value="Natal">Natal</option>
              <option value="Campo Grande">Campo Grande</option>
              <option value="João Pessoa">João Pessoa</option>
              <option value="Aracaju">Aracaju</option>
              <option value="Teresina">Teresina</option>
              <option value="Cuiabá">Cuiabá</option>
              <option value="Macapá">Macapá</option>
              <option value="Rio Branco">Rio Branco</option>
              <option value="Boa Vista">Boa Vista</option>
              <option value="Palmas">Palmas</option>
              <option value="Florianópolis">Florianópolis</option>
              <option value="Porto Velho">Porto Velho</option>
            </select>
          </div>
          <div className="col-md-4">
            <label className="form-label">Perfil do Cliente:</label>
            <select
              name="perfil"
              value={formData.perfil}
              onChange={handleChange}
              className="form-select"
              style={{ borderRadius: "8px" }}
            >
              <option value="Aposentado">Aposentado</option>
              <option value="Estudante">Estudante</option>
              <option value="Não Aposentado">Não Aposentado</option>
            </select>
          </div>
          <div className="col-md-4">
            <label className="form-label">Modelo de Classificação:</label>
            <select
              name="modelo_selecionado"
              value={formData.modelo_selecionado}
              onChange={handleChange}
              className="form-select"
              style={{ borderRadius: "8px" }}
            >
              <option value="Clássico">Clássico</option>
              <option value="Quântico">Quântico</option>
            </select>
          </div>
        </div>

        <button
          type="submit"
          className="btn btn-primary w-100 mt-4"
          style={{ borderRadius: "10px", padding: "10px" }}
        >
          Classificar Transação
        </button>
        {/* Rodapé */}
        <div style={{ textAlign: "center", color: "black", marginTop: "40px" }}>
          <p style={{ marginBottom: "0px" }}>
            Desenvolvido por Weslley Rosa Prado
          </p>
          <p style={{ marginBottom: "0px" }}>
            Orientador: Professor Doutor José Alexandre Nogueira
          </p>
          <p style={{ marginBottom: "0px" }}>
            Projeto Prático Baseado no Trabalho de Conclusão de Curso de Física
          </p>
          <p style={{ marginBottom: "0px" }}>
            Universidade Federal do Espírito Santo - UFES
          </p>
        </div>
      </form>

      {resultado && (
        <div
          className="mt-4 alert alert-success text-center"
          style={{ borderRadius: "8px" }}
        >
          {resultado}
        </div>
      )}
      {erro && (
        <div
          className="mt-4 alert alert-danger text-center"
          style={{ borderRadius: "8px" }}
        >
          {erro}
        </div>
      )}
    </div>
  );
}
