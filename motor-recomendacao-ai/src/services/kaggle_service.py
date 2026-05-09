import os
import kagglehub
import shutil
from dotenv import load_dotenv

# Isso faz a aplicação ler o arquivo .env que você criou
load_dotenv()


def baixar_dataset(dataset_handle):
    # O kagglehub agora encontrará o KAGGLE_API_TOKEN carregado pelo load_dotenv()
    print(f"Buscando dataset: {dataset_handle}...")

    try:
        path_origem = kagglehub.dataset_download(dataset_handle)

        # Caminho absoluto para a pasta /data
        path_destino = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '../../data'))

        if not os.path.exists(path_destino):
            os.makedirs(path_destino)

        for arquivo in os.listdir(path_origem):
            shutil.copy2(os.path.join(path_origem, arquivo),
                         os.path.join(path_destino, arquivo))

        print(f"Dados atualizados com sucesso em /data")
        return path_destino
    except Exception as e:
        print(f"Erro na autenticação ou download: {e}")
        return None


if __name__ == "__main__":
    # Teste de carga
    baixar_dataset("darpan25bajaj/events-datasetfor-collaborative-filtering")
