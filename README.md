```
# PromoIngresso - Motor de Recomendação AI

Este repositório contém a Inteligência Artificial e a API RESTful do projeto **PromoIngresso**. O sistema atua como
o "cérebro" analítico da aplicação, operando um motor de recomendação híbrido e multicontextual que sugere eventos
aos usuários com base em similaridade de interesses e viabilidade física ou digital.

O motor utiliza o algoritmo de filtragem colaborativa **k-Nearest Neighbors (k-NN)** treinado com a métrica de
 **Similaridade do Cosseno**, e aplica uma rigorosa lógica de pós-filtragem contextual utilizando a fórmula de
 **Haversine** para calcular distâncias. 

A base de dados utilizada é o dataset oficial do *Meetup* (`megelon/meetup`), baixada de forma automatizada via
integração com a plataforma Kaggle.

---

## 🛠️ Tecnologias Utilizadas
* **Python 3.x**
* **Flask & Flask-CORS** (Para a construção e disponibilização da API REST)
* **Scikit-Learn** (Para o algoritmo de Machine Learning k-NN)
* **Pandas & NumPy** (Para manipulação de dados e conversões trigonométricas)
* **SciPy** (Para conversão da base de dados em Matrizes Esparsas otimizadas)
* **Kagglehub & Python-dotenv** (Para download dinâmico do dataset e gerenciamento de variáveis de ambiente)

---

## ⚙️ Instruções de Instalação e Configuração

```
### 1. Clonar o Repositório
```bash
git clone https://github.com/seu-usuario/PromoIngressoAi.git
cd PromoIngressoAi
```

### 2. Configurar o Ambiente Virtual (Recomendado)
Crie e ative um ambiente virtual para isolar as dependências do projeto:
```bash
# No Windows:
python -m venv venv
venv\Scripts\activate

# No Linux/Mac:
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar as Dependências
Como o sistema faz uso de bibliotecas matemáticas e de rotas *web*, instale as dependências via `pip`:
```bash
pip install flask flask-cors pandas numpy scikit-learn scipy kagglehub python-dotenv
```

### 4. Configuração das Variáveis de Ambiente
O sistema utiliza o pacote `dotenv` para gerenciar chaves. Na raiz do projeto, crie um arquivo chamado `.env` 
e insira suas credenciais do Kaggle (necessário para baixar a base de dados na primeira execução):
```env
KAGGLE_USERNAME="seu_usuario_do_kaggle"
KAGGLE_KEY="sua_chave_api_do_kaggle"
```

---

## 🚀 Como Executar o Sistema

Para rodar o motor de IA, execute o script principal na raiz do projeto:
```bash
python app.py
```

O sistema executará automaticamente três etapas de inicialização:
1. **[1/3] Carregamento de Dados:** Valida se a base local existe; caso contrário, realiza o download via
*Kagglehub*, cruza as informações e gera a matriz usuário-tópico.
3. **[2/3] Treinamento:** Processa a matriz esparsa e treina a IA com o algoritmo k-NN.
4. **[3/3] Servidor REST:** Inicializa o servidor web Flask na porta 5000 e libera a API para escuta.

Para facilitar os testes, o console imprimirá uma lista randômica com exemplos de IDs de usuários reais
validados pela base de dados que você pode utilizar na URL.

---

## 📡 Uso da API (Endpoints)

Com o servidor rodando, a API disponibilizará a rota principal para consulta de recomendações:

**Endpoint:**
`GET http://localhost:5000/api/recomendacoes/<ID_DO_USUARIO>`

**Descrição:** 
Gera um ranqueamento inteligente e retorna os 10 eventos mais recomendados para o usuário informado. 
O processamento interno avalia similaridade social, filtros de popularidade e bônus/penalidades de contexto
geográfico ou viés de evento *online*.

**Exemplo de Resposta (JSON):**
```json
{
  "usuario_id": 12345,
  "recomendacoes": [
    {
      "event_id": "987654321",
      "event_name": "Tech Meetup São Paulo",
      "venue_name": "Centro de Convenções",
      "venue_city": "São Paulo",
      "group_rating": 4.8,
      "venue_rating": 4.5,
      "yes_rsvp_count": 120
    }
  ]
}
```
*Nota: Caso o usuário não seja encontrado ou ocorra algum erro de tráfego, o sistema retornará o status
de erro `404` ou `500` acompanhado de uma mensagem JSON descritiva*.

---

## 🧪 Rodando os Testes

O projeto foi construído sob uma sólida engenharia de *software* com validações unitárias e de
integração implementadas através da biblioteca `unittest` nativa do Python. 

Para validar matematicamente os cálculos de distância (Haversine):
```
bash
python -m unittest tests/test_geo.py
```

Para validar a integração completa do Motor de IA simulando o acesso a um usuário real e prevenindo 
erros para usuários falsos:
```bash
python -m unittest tests/test_integration.py
```
