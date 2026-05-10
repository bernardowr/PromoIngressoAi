import math
import random
from src.services.kaggle_service import obter_diretorio_dados
import os
import sys
from pathlib import Path

from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def calcular_distancia_km(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return float('inf')
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * \
        math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


PROJETO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJETO_ROOT))

# =================================================================
# FASE 1: INGESTÃO E PRÉ-PROCESSAMENTO DE DADOS
# =================================================================
print("1. Buscando base de dados do Meetup...")
path = obter_diretorio_dados()
if path is None:
    raise RuntimeError(
        "Não foi possível preparar os dados do Meetup. Verifique o serviço de download.")
members_topics_path = os.path.join(path, 'members_topics.csv')

# Carrega os dados de interesses dos membros
df_members_topics = pd.read_csv(members_topics_path)
df_members_topics_clean = df_members_topics.drop_duplicates(
    subset=['member_id', 'topic_id']).copy()

# AMOSTRAGEM PARA REDUZIR TAMANHO: Usa apenas 10% dos dados para evitar problemas de memória
df_members_topics_clean = df_members_topics_clean.sample(
    frac=0.1, random_state=42)

print(
    f"Total de interesses diretos dos membros carregados (após amostragem): {len(df_members_topics_clean)}")

# =================================================================
# CRUZAMENTO DE DADOS: ENRIQUECER INTERESSES COM GRUPOS E CONTEXTO LOCAL
# =================================================================
print("3. Enriquecendo interesses com dados de grupos e localização...")

# Carrega dados de membros e seus grupos + localização
members_path = os.path.join(path, 'members.csv')
df_members = pd.read_csv(
    members_path,
    encoding='latin1',
    usecols=['member_id', 'group_id', 'city', 'country', 'lat', 'lon']
)
df_members_clean = df_members.drop_duplicates().copy()
df_members_clean = df_members_clean.sample(frac=0.1, random_state=42)

# Mapa de localização dos usuários para recomendação geográfica
print("3.1 Carregando localização de membros...")
df_member_locations = pd.read_csv(
    members_path,
    encoding='latin1',
    usecols=['member_id', 'city', 'country', 'lat', 'lon']
).drop_duplicates(subset=['member_id']).set_index('member_id')

# Carrega tópicos dos grupos
groups_topics_path = os.path.join(path, 'groups_topics.csv')
df_groups_topics = pd.read_csv(groups_topics_path, encoding='latin1')
df_groups_topics_clean = df_groups_topics.drop_duplicates().copy()

# Junta membros com tópicos de seus grupos
df_member_group_topics = pd.merge(
    df_members_clean,
    df_groups_topics_clean,
    on='group_id',
    how='inner'
)[['member_id', 'topic_id']]

# Remove duplicatas
df_member_group_topics = df_member_group_topics.drop_duplicates()

# Carrega eventos e une com grupos que têm tópico
events_path = os.path.join(path, 'events.csv')
df_events = pd.read_csv(
    events_path,
    encoding='latin1',
    usecols=[
        'event_id', 'group_id', 'event_name', 'venue_id', 'venue.city',
        'venue.country', 'venue.lat', 'venue.lon', 'yes_rsvp_count'
    ]
)
df_events_clean = df_events.drop_duplicates(subset=['event_id']).copy()

# Enriquecimento com dados de venue
venues_path = os.path.join(path, 'venues.csv')
df_venues = pd.read_csv(
    venues_path,
    encoding='latin1',
    usecols=['venue_id', 'venue_name', 'rating',
             'normalised_rating', 'city', 'country', 'lat', 'lon']
)
df_venues_clean = df_venues.drop_duplicates(subset=['venue_id']).copy()

# Prioriza valores do evento sobre dados de venue quando existentes
# Usa a tabela de venues para preencher local e rating
if 'venue_id' in df_events_clean.columns:
    df_events_clean = pd.merge(
        df_events_clean,
        df_venues_clean,
        on='venue_id',
        how='left',
        suffixes=('', '_venue')
    )
else:
    df_events_clean['venue_name'] = None
    df_events_clean['rating'] = 0
    df_events_clean['normalised_rating'] = 0

# Ajusta colunas de localização com fallback dos venues
for col in ['venue.city', 'venue.country', 'venue.lat', 'venue.lon']:
    if col not in df_events_clean.columns:
        df_events_clean[col] = None

df_events_clean['venue_rating'] = df_events_clean['normalised_rating'].fillna(
    0)

df_event_topics = pd.merge(
    df_events_clean,
    df_groups_topics_clean[['group_id', 'topic_id']],
    on='group_id',
    how='inner'
)

groups_path = os.path.join(path, 'groups.csv')
df_groups = pd.read_csv(
    groups_path,
    encoding='latin1',
    usecols=['group_id', 'rating']
)

df_event_topics = pd.merge(
    df_event_topics,
    df_groups.drop_duplicates(subset=['group_id']).rename(
        columns={'rating': 'group_rating'}),
    on='group_id',
    how='left'
)

df_event_topics['group_rating'] = df_event_topics['group_rating'].fillna(0)
df_event_topics['yes_rsvp_count'] = df_event_topics['yes_rsvp_count'].fillna(0)
df_event_topics['event_score'] = (
    df_event_topics['group_rating'] * 2 +
    df_event_topics['venue_rating'] * 1.5 +
    df_event_topics['yes_rsvp_count'] / 40
)

# Combina interesses diretos com interesses dos grupos
# + casos contextuais de eventos próximos e populares
print("3.2 Combinando interesses diretos e de grupo...")
df_all_interests = pd.concat([
    df_members_topics_clean[['member_id', 'topic_id']],
    df_member_group_topics
]).drop_duplicates()

print(
    f"Total de interesses enriquecidos (diretos + grupos): {len(df_all_interests)}")

# Cria a Matriz Usuário-Tópico (pivot table) com 1 para interesse presente, 0 para ausente
print("4. Construindo a Matriz Usuário-Tópico...")
matriz_usuario_topico = df_all_interests.pivot_table(
    index='member_id', columns='topic_id', aggfunc='size', fill_value=0)
# Converte para binário: 1 se tem interesse, 0 se não
matriz_usuario_topico = (matriz_usuario_topico > 0).astype(int)

# NORMALIZAÇÃO: Essencial para que o cálculo de distância do k-NN não seja distorcido
scaler = MinMaxScaler()
matriz_normalizada = scaler.fit_transform(matriz_usuario_topico)

# Recria o DataFrame com os IDs originais preservados
df_matriz_norm = pd.DataFrame(
    matriz_normalizada,
    index=matriz_usuario_topico.index,
    columns=matriz_usuario_topico.columns
)
print("Matriz criada e normalizada com sucesso!")

# =================================================================
# FASE 2: MOTOR DE INTELIGÊNCIA ARTIFICIAL
# =================================================================
# Agrupamento (Clustering) com K-Means
print("5. Aplicando K-Means para segmentar os usuários por interesses...")
kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
df_matriz_norm['cluster'] = kmeans.fit_predict(df_matriz_norm)

# Filtragem Colaborativa com k-Nearest Neighbors (k-NN)
print("6. Treinando o modelo k-NN (k=20 vizinhos mais próximos)...")
knn = NearestNeighbors(n_neighbors=20, metric='euclidean')
# Treinamos o k-NN com os dados normalizados (ignorando a coluna do K-Means)
knn.fit(df_matriz_norm.drop('cluster', axis=1))

print("\n--- TREINAMENTO CONCLUÍDO ---")
print("O sistema está pronto para gerar recomendações baseadas na similaridade de interesses.")

# =================================================================
# FASE 2.1: GERANDO A RECOMENDAÇÃO (TESTE PRÁTICO)
# =================================================================


def gerar_recomendacao(usuario_id, modelo_knn, df_normalizada, matriz_original):
    # Verifica se o usuário existe na base
    if usuario_id not in df_normalizada.index:
        return "Usuário não encontrado."

    # 1. Isola os dados do usuário alvo (ignorando a coluna 'cluster' do K-Means)
    dados_usuario = df_normalizada.loc[usuario_id].drop(
        'cluster').values.reshape(1, -1)

    # 2. O k-NN busca as instâncias mais similares
    distancias, indices = modelo_knn.kneighbors(dados_usuario)

    # Extrai os IDs reais dos vizinhos encontrados
    vizinhos_ids = df_normalizada.iloc[indices[0]].index.tolist()
    if usuario_id in vizinhos_ids:
        vizinhos_ids.remove(usuario_id)

    # 3. Identifica tópicos do usuário e dos vizinhos
    topicos_usuario = matriz_original.loc[usuario_id]
    topicos_usuario = topicos_usuario[topicos_usuario == 1].index.tolist()

    topicos_vizinhos = []
    for vizinho in vizinhos_ids:
        topicos_vizinho = matriz_original.loc[vizinho]
        topicos_vizinhos.extend(
            topicos_vizinho[topicos_vizinho == 1].index.tolist())

    from collections import Counter
    contador_topicos = Counter(topicos_vizinhos)
    topicos_relevantes = [topic for topic,
                          count in contador_topicos.items() if count >= 2]

    # 4. Busca eventos relacionados a tópicos relevantes e grupos dos vizinhos
    grupos_vizinhos = df_members_clean[df_members_clean['member_id'].isin(
        vizinhos_ids)]['group_id'].unique()
    eventos_candidatos = df_event_topics[
        (df_event_topics['topic_id'].isin(topicos_relevantes)) |
        (df_event_topics['group_id'].isin(grupos_vizinhos))
    ].copy()

    # 5. Localização do usuário para priorizar eventos próximos
    if usuario_id in df_member_locations.index:
        loc = df_member_locations.loc[usuario_id]
        user_city = str(loc['city']).strip().lower()
        user_country = str(loc['country']).strip().lower()
        user_lat = loc['lat']
        user_lon = loc['lon']
    else:
        user_city = None
        user_country = None
        user_lat = None
        user_lon = None

    def event_distance(row):
        if user_lat is None or user_lon is None:
            return float('inf')
        return calcular_distancia_km(user_lat, user_lon, row['venue.lat'], row['venue.lon'])

    eventos_candidatos['same_city'] = eventos_candidatos['venue.city'].fillna(
        '').str.lower() == user_city
    eventos_candidatos['distance_km'] = eventos_candidatos.apply(
        event_distance, axis=1)
    eventos_candidatos['same_location'] = (
        eventos_candidatos['same_city'] | (
            eventos_candidatos['distance_km'] <= 50)
    )

    eventos_mesma_cidade = eventos_candidatos[eventos_candidatos['same_city']].copy(
    )
    eventos_mesma_cidade = eventos_mesma_cidade[
        (eventos_mesma_cidade['group_rating'] >= 4.0) &
        (eventos_mesma_cidade['venue_rating'] >= 3.5) &
        (eventos_mesma_cidade['yes_rsvp_count'] >= 10)
    ]

    if not eventos_mesma_cidade.empty:
        eventos_candidatos = eventos_mesma_cidade
    else:
        eventos_mesma_localizacao = eventos_candidatos[eventos_candidatos['same_location']].copy(
        )
        eventos_mesma_localizacao = eventos_mesma_localizacao[
            (eventos_mesma_localizacao['group_rating'] >= 4.0) &
            (eventos_mesma_localizacao['venue_rating'] >= 3.5) &
            (eventos_mesma_localizacao['yes_rsvp_count'] >= 10)
        ]

        if not eventos_mesma_localizacao.empty:
            eventos_candidatos = eventos_mesma_localizacao
        else:
            print("Nenhum evento na mesma cidade ou localização próxima encontrado; usando fallback por popularidade.")
            eventos_candidatos = df_event_topics[
                (df_event_topics['group_rating'] >= 4.0) &
                (df_event_topics['venue_rating'] >= 3.5) &
                (df_event_topics['yes_rsvp_count'] >= 10)
            ].copy()
            eventos_candidatos['distance_km'] = eventos_candidatos.apply(
                event_distance, axis=1)
            eventos_candidatos['same_city'] = eventos_candidatos['venue.city'].fillna(
                '').str.lower() == user_city

    eventos_agrupados = eventos_candidatos.groupby(
        ['event_id', 'event_name', 'group_id', 'venue_name', 'venue.city',
            'venue.country', 'venue.lat', 'venue.lon'],
        as_index=False
    ).agg(
        group_rating=('group_rating', 'max'),
        venue_rating=('venue_rating', 'max'),
        yes_rsvp_count=('yes_rsvp_count', 'max'),
        topic_match_count=('topic_id', 'nunique'),
        same_city=('same_city', 'max'),
        distance_km=('distance_km', 'min')
    )

    eventos_agrupados['group_bonus'] = eventos_agrupados['group_id'].isin(
        grupos_vizinhos).astype(int) * 3
    eventos_agrupados['score'] = (
        eventos_agrupados['group_rating'] * 3 +
        eventos_agrupados['venue_rating'] * 2 +
        eventos_agrupados['yes_rsvp_count'] / 40 +
        eventos_agrupados['topic_match_count'] * 1.5 +
        eventos_agrupados['same_city'] * 5 +
        eventos_agrupados['group_bonus'] -
        eventos_agrupados['distance_km'] / 20
    )

    eventos_ordenados = eventos_agrupados.sort_values(
        by='score', ascending=False)
    top_eventos = eventos_ordenados.head(10)

    recomendacoes = [
        {
            "event_id": str(row['event_id']),
            "event_name": row['event_name'],
            "venue_name": row['venue_name'] if 'venue_name' in row else None,
            "venue_city": row['venue.city'] if 'venue.city' in row else None,
            "group_rating": float(row['group_rating']),
            "venue_rating": float(row['venue_rating']),
            "yes_rsvp_count": int(row['yes_rsvp_count'])
        }
        for _, row in top_eventos.iterrows()
    ]

    # Os prints agora funcionam como logs no terminal do servidor
    print(f"\n[LOG] Requisição processada para o usuário {usuario_id}:")
    print(f"      - Tópicos de interesse diretos: {len(topicos_usuario)}")
    print(
        f"      - Tópicos relevantes via vizinhos k-NN: {len(topicos_relevantes)}")
    print(
        f"      - Eventos candidatos após filtros: {len(eventos_candidatos)}")
    print(f"      - Recomendações retornadas: {len(recomendacoes)}")
    print("-" * 60)

    return recomendacoes


app = Flask(__name__)
CORS(app)  # Habilita acesso do React (ou outras origens) à nossa API Flask


@app.route('/api/recomendacoes/<int:usuario_id>', methods=['GET'])
def api_recomendacoes(usuario_id):
    """Endpoint RESTful para retornar as recomendações de um usuário em JSON"""
    try:
        recomendacoes = gerar_recomendacao(
            usuario_id, knn, df_matriz_norm, matriz_usuario_topico)
        if isinstance(recomendacoes, str):  # Retorno de erro caso o "Usuário não encontrado."
            return jsonify({"erro": recomendacoes}), 404
        return jsonify({"usuario_id": usuario_id, "recomendacoes": recomendacoes}), 200
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


if __name__ == '__main__':
    print("\n🚀 Iniciando servidor REST da API de Recomendação na porta 5000...")
    print("Pode ser acessada em: http://localhost:5000/api/recomendacoes/<ID_DO_USUARIO>")
    app.run(host='0.0.0.0', port=5000, debug=False)
