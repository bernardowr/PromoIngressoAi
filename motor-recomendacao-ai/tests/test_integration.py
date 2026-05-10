from src.engine.recommender import treinar_modelo, gerar_recomendacao
from src.data.pipeline import preparar_dados
import unittest
import warnings
import sys
from pathlib import Path

# Garante que a raiz do projeto esteja no path para o Python encontrar a pasta 'src'
PROJETO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJETO_ROOT))


class TestMotorRecomendacaoReal(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Roda apenas 1 vez ANTES dos testes começarem.
        Carrega a base real do Kaggle e treina o modelo para ser usado nos testes.
        """
        # Ignora avisos (warnings) de versões do Pandas para manter o terminal limpo no teste
        warnings.filterwarnings("ignore")

        print("\n[TESTES] Carregando a base de dados real do Meetup...")
        cls.dados = preparar_dados()

        print("\n[TESTES] Treinando a Inteligência Artificial...")
        cls.modelo_knn = treinar_modelo(cls.dados['matriz_esparsa'])

        # Extrai todos os IDs válidos para usarmos nos testes
        cls.ids_validos = cls.dados['df_matriz_norm'].index.tolist()

    def test_pipeline_carregou_dados_corretamente(self):
        """Garante que a matriz de usuários não está vazia."""
        self.assertGreater(len(self.ids_validos), 0,
                           "A base de dados real falhou ao carregar ou está vazia.")
        self.assertIn('matriz_esparsa', self.dados)

    def test_recomendacao_para_usuario_real(self):
        """Pega o primeiro usuário real da base e testa se o motor cospe recomendações válidas."""
        usuario_teste = self.ids_validos[0]
        recomendacoes = gerar_recomendacao(
            usuario_teste, self.modelo_knn, self.dados)

        self.assertIsInstance(
            recomendacoes, list, "A recomendação deve retornar uma lista de dicionários.")
        self.assertTrue(len(recomendacoes) <= 10,
                        "O motor não deve retornar mais do que 10 eventos.")

        if len(recomendacoes) > 0:
            primeiro_evento = recomendacoes[0]
            self.assertIn('event_id', primeiro_evento,
                          "O evento recomendado não possui ID.")
            self.assertIn('event_name', primeiro_evento,
                          "O evento recomendado não possui Nome.")
            self.assertIn('group_rating', primeiro_evento,
                          "O evento não retornou a nota do grupo.")

    def test_recomendacao_usuario_inexistente(self):
        """Garante que a API não quebre (Erro 500) se alguem passar um ID falso."""
        usuario_ficticio = -999999999
        recomendacoes = gerar_recomendacao(
            usuario_ficticio, self.modelo_knn, self.dados)
        self.assertEqual(recomendacoes, "Usuário não encontrado.")


if __name__ == '__main__':
    unittest.main()
