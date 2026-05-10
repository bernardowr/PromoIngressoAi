import random
from src.data.pipeline import preparar_dados
from src.engine.recommender import treinar_modelo
from src.api.routes import criar_app

if __name__ == '__main__':
    print("\n=== MOTOR DE RECOMENDAÇÃO AI ===")

    print("\n[1/3] Iniciando carregamento e processamento de dados (Pandas)...")
    dados = preparar_dados()

    print("\n[2/3] Iniciando treinamento da Inteligência Artificial (K-NN)...")
    modelo_knn = treinar_modelo(dados['matriz_esparsa'])

    print("\n[3/3] Iniciando servidor REST da API na porta 5000...")
    app = criar_app(modelo_knn, dados)

    # Extrai alguns exemplos de IDs da base completa para facilitar testes rápidos (copiar/colar)
    ids_validos = dados['df_matriz_norm'].index.tolist()
    exemplos_ids = random.sample(ids_validos, min(10, len(ids_validos)))
    print(
        f"👉 DICA: (Exemplos randômicos de IDs para testar na URL): {exemplos_ids}")
    print("Acesse no navegador: http://localhost:5000/api/recomendacoes/<ID_DO_USUARIO>")

    app.run(host='0.0.0.0', port=5000, debug=False)
