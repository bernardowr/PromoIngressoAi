from src.utils.geo import calcular_distancia_km
import unittest
import sys
from pathlib import Path

# Garante que a raiz do projeto esteja no path para o Python encontrar a pasta 'src'
PROJETO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJETO_ROOT))


class TestGeo(unittest.TestCase):

    def test_distancia_mesmo_ponto(self):
        """A distância entre exatamente o mesmo ponto deve ser 0 km."""
        distancia = calcular_distancia_km(-23.5505, -
                                          46.6333, -23.5505, -46.6333)
        self.assertEqual(distancia, 0)

    def test_distancia_sao_paulo_rio(self):
        """Testa se a fórmula calcula corretamente a distância de SP ao RJ (~360km)."""
        distancia = calcular_distancia_km(-23.5505, -
                                          46.6333, -22.9068, -43.1729)
        # A distância em linha reta fica em torno de 350 a 400 km dependendo do ponto exato
        self.assertTrue(350 < distancia < 400,
                        f"Distância calculada foi: {distancia}")


if __name__ == '__main__':
    unittest.main()
