from flask import Flask, jsonify
from flask_cors import CORS
from src.engine.recommender import gerar_recomendacao


def criar_app(modelo_knn, dados):
    """Cria e configura a instância do servidor Flask."""
    app = Flask(__name__)
    CORS(app)

    @app.route('/api/recomendacoes/<int:usuario_id>', methods=['GET'])
    def api_recomendacoes(usuario_id):
        """Endpoint RESTful para retornar as recomendações de um usuário em JSON."""
        try:
            recomendacoes = gerar_recomendacao(usuario_id, modelo_knn, dados)
            if isinstance(recomendacoes, str):
                return jsonify({"erro": recomendacoes}), 404
            return jsonify({"usuario_id": usuario_id, "recomendacoes": recomendacoes}), 200
        except Exception as e:
            return jsonify({"erro": str(e)}), 500

    return app
