import pandas as pd
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from src.utils.geo import calcular_distancia_km


def treinar_modelo(matriz_esparsa):
    print("6. Treinando o modelo k-NN (k=20 vizinhos mais próximos)...")
    knn = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute')
    knn.fit(matriz_esparsa)
    print("\n--- TREINAMENTO CONCLUÍDO ---")
    print("O sistema está pronto para gerar recomendações baseadas na similaridade de interesses.")
    return knn


def gerar_recomendacao(usuario_id, modelo_knn, dados):
    df_normalizada = dados['df_matriz_norm']
    matriz_original = dados['matriz_usuario_topico']
    df_members_clean = dados['df_members_clean']
    df_event_topics = dados['df_event_topics']
    df_member_locations = dados['df_member_locations']

    if usuario_id not in df_normalizada.index:
        return "Usuário não encontrado."

    dados_usuario = df_normalizada.loc[usuario_id].values.reshape(1, -1)
    distancias, indices = modelo_knn.kneighbors(dados_usuario)

    vizinhos_ids = df_normalizada.iloc[indices[0]].index.tolist()
    if usuario_id in vizinhos_ids:
        vizinhos_ids.remove(usuario_id)

    # Otimização: Vetorização com Pandas para extrair tópicos relevantes sem usar loops (for)
    # Pega apenas as linhas dos vizinhos e soma a ocorrência de cada tópico
    matriz_vizinhos = matriz_original.loc[vizinhos_ids]
    contagem_topicos = matriz_vizinhos.sum(axis=0)
    # Filtra tópicos que aparecem para 2 ou mais vizinhos
    topicos_relevantes = contagem_topicos[contagem_topicos >= 2].index.tolist()

    grupos_vizinhos = df_members_clean[df_members_clean['member_id'].isin(
        vizinhos_ids)]['group_id'].unique()
    eventos_candidatos = df_event_topics[
        (df_event_topics['topic_id'].isin(topicos_relevantes)) |
        (df_event_topics['group_id'].isin(grupos_vizinhos))
    ].copy()

    if usuario_id in df_member_locations.index:
        loc = df_member_locations.loc[usuario_id]
        user_city = str(loc['city']).strip().lower()
        user_lat = loc['lat']
        user_lon = loc['lon']
    else:
        user_city = None
        user_lat = None
        user_lon = None

    def aplicar_filtros_geograficos(df_alvo):
        df_alvo['same_city'] = df_alvo['venue.city'].fillna(
            '').str.lower() == user_city
        if user_lat is None or user_lon is None:
            df_alvo['distance_km'] = float('inf')
        else:
            df_alvo['distance_km'] = calcular_distancia_km(
                user_lat, user_lon, df_alvo['venue.lat'], df_alvo['venue.lon']).fillna(float('inf'))

        # Eventos sem coordenadas de venue (como eventos online) recebem passe livre de localidade
        df_alvo['is_online'] = df_alvo['venue.lat'].isna()
        df_alvo['same_location'] = df_alvo['same_city'] | (
            df_alvo['distance_km'] <= 50) | df_alvo['is_online']
        return df_alvo

    eventos_candidatos = aplicar_filtros_geograficos(eventos_candidatos)

    mascara_qualidade = (
        (eventos_candidatos['group_rating'] >= 4.0) &
        ((eventos_candidatos['venue_rating'] >= 3.5) | (eventos_candidatos['venue_rating'].fillna(0) == 0)) &
        (eventos_candidatos['yes_rsvp_count'] >= 10)
    )
    eventos_qualificados = eventos_candidatos[mascara_qualidade]

    eventos_filtrados = eventos_qualificados[eventos_qualificados['same_city']].copy(
    )

    if eventos_filtrados.empty:
        eventos_filtrados = eventos_qualificados[eventos_qualificados['same_location']].copy(
        )

    if eventos_filtrados.empty:
        print("Nenhum evento próximo encontrado; usando fallback por popularidade.")
        df_populares = df_event_topics[
            (df_event_topics['group_rating'] >= 4.0) &
            ((df_event_topics['venue_rating'] >= 3.5) | (df_event_topics['venue_rating'].fillna(0) == 0)) &
            (df_event_topics['yes_rsvp_count'] >= 10)
        ].copy()
        eventos_filtrados = aplicar_filtros_geograficos(df_populares)

    # Previne que a API quebre se o fallback também não retornar nenhum evento
    if eventos_filtrados.empty:
        return []

    eventos_agrupados = eventos_filtrados.groupby(
        ['event_id', 'event_name', 'group_id', 'venue_name',
            'venue.city', 'venue.country', 'venue.lat', 'venue.lon'],
        as_index=False, dropna=False
    ).agg(
        group_rating=('group_rating', 'max'), venue_rating=('venue_rating', 'max'),
        yes_rsvp_count=('yes_rsvp_count', 'max'), topic_match_count=('topic_id', 'nunique'),
        same_city=('same_city', 'max'), distance_km=('distance_km', 'min')
    )

    eventos_agrupados['group_bonus'] = eventos_agrupados['group_id'].isin(
        grupos_vizinhos).astype(int) * 3

    eventos_agrupados['is_online'] = eventos_agrupados['venue.lat'].isna()
    pontuacao_venue = eventos_agrupados['venue_rating'].replace(0, 3.5) * 2
    penalidade_dist = eventos_agrupados['distance_km'].replace(
        float('inf'), 0) / 20

    eventos_agrupados['score'] = (
        eventos_agrupados['group_rating'] * 3 + pontuacao_venue +
        eventos_agrupados['yes_rsvp_count'] / 40 + eventos_agrupados['topic_match_count'] * 1.5 +
        eventos_agrupados['same_city'] * 5 + eventos_agrupados['group_bonus'] +
        (eventos_agrupados['is_online'].astype(int) * 5) - penalidade_dist
    )

    top_eventos = eventos_agrupados.sort_values(
        by='score', ascending=False).head(10)

    recomendacoes = [{
        "event_id": str(row['event_id']), "event_name": row['event_name'],
        "venue_name": row['venue_name'] if pd.notna(row['venue_name']) else None,
        "venue_city": row['venue.city'] if pd.notna(row['venue.city']) else None,
        "group_rating": float(row['group_rating']), "venue_rating": float(row['venue_rating']),
        "yes_rsvp_count": int(row['yes_rsvp_count'])
    } for _, row in top_eventos.iterrows()]

    return recomendacoes
