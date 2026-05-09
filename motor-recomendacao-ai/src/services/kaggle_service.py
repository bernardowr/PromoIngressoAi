import os
import shutil
import kagglehub
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

CAMINHO_RELATIVO_DADOS = '../../data'
DATASET_MEETUP = 'megelon/meetup'


def obter_caminho_diretorio_dados() -> str:
    """Retorna o caminho absoluto para o diretório de dados, criando-o se necessário."""
    diretorio_dados = os.path.abspath(os.path.join(
        os.path.dirname(__file__), CAMINHO_RELATIVO_DADOS))
    os.makedirs(diretorio_dados, exist_ok=True)
    return diretorio_dados


def baixar_meetup_dataset(dataset_handle: str = DATASET_MEETUP) -> str | None:
    """Baixa o dataset do Kaggle e copia os arquivos para a pasta local de dados."""
    print(f"Baixando dataset de eventos: {dataset_handle}...")

    try:
        caminho_origem = kagglehub.dataset_download(dataset_handle)
    except Exception as e:
        print(f"Erro no download do Kaggle: {e}")
        return None

    diretorio_dados = obter_caminho_diretorio_dados()

    for arquivo in os.listdir(caminho_origem):
        origem = os.path.join(caminho_origem, arquivo)
        destino = os.path.join(diretorio_dados, arquivo)
        shutil.copy2(origem, destino)

    print(f"Dados copiados para {diretorio_dados}")
    return diretorio_dados


def obter_diretorio_dados() -> str | None:
    """Retorna o caminho do diretório de dados se ele já contiver arquivos ou tenta baixá-los."""
    diretorio_dados = obter_caminho_diretorio_dados()
    arquivos = os.listdir(diretorio_dados)

    if arquivos:
        print(f"Arquivos de dados encontrados em {diretorio_dados}: {arquivos}")
        return diretorio_dados

    return baixar_meetup_dataset()


if __name__ == "__main__":
    caminho_dados = obter_diretorio_dados()
    if caminho_dados:
        print(f"Diretório de dados pronto: {caminho_dados}")
    else:
        print("Não foi possível preparar o diretório de dados.")
