import requests
import spacy
import heapq
from collections import deque
import time
from sklearn.metrics.pairwise import cosine_similarity
from settings import Settings, get_settings
from fastapi import APIRouter, HTTPException, Depends
from typing import Annotated

router = APIRouter()
nlp = spacy.load('en_core_web_lg')

def find_related_titles(
    anime, 
    settings: Settings
):
    '''Use BFS to find all related anime titles for a given anime and return them in a set'''
    related_titles = set()
    queue = deque()

    related_titles.add(anime['title'])
    for r_anime in anime['related_anime']:
        queue.append(r_anime['node'])

    while queue:
        current_anime = queue.popleft()
        title = current_anime['title']
        if title not in related_titles:
            related_titles.add(title)
            #Fetch additional related anime from the API
            try: 
                response = requests.get(
                    f'https://api.myanimelist.net/v2/anime/{current_anime['id']}?fields=related_anime', 
                    headers={'X-MAL-Client-ID': settings.client_id}
                )
                response.headers['Content-Type'] = 'application/json'
                anime_details = response.json()
                for r_anime in anime_details['related_anime']:
                    queue.append(r_anime['node'])
            except requests.RequestException as e:
                print(f'Error fetching related anime for {title}: {e}')
    return related_titles

def find_recommendations(similarities, num_of_good, num_of_bad, rank_data, related_titles):
    '''Use a heap to find the n most similar and m least similar anime to the selected anime'''
    related = set(related_titles)
    most_similar = heapq.nlargest(num_of_good + len(related_titles), enumerate(similarities), key=lambda x: x[1])
    most_similar_indexes = [index for index, _ in most_similar if rank_data[index]['node']['title'] not in related_titles][:num_of_good]

    for i in most_similar_indexes:
        related.add(rank_data[i]['node']['title'])

    least_similar = heapq.nsmallest(num_of_bad + len(related), enumerate(similarities), key=lambda x: x[1])
    least_similar_indexes = [index for index, _ in least_similar if rank_data[index]['node']['title'] not in related_titles][:num_of_bad]

    return most_similar_indexes + least_similar_indexes
        
@router.get("/search-anime")
def search_anime(
    query: str,
    settings: Annotated[Settings, Depends(get_settings)]
):
    response = requests.get(
        f'https://api.myanimelist.net/v2/anime?q={query}&limit=100&fields=id,title,main_picture', 
        headers={'X-MAL-Client-ID': settings.client_id}
    )
    response.headers['Content-Type'] = 'application/json'

    if response.status_code == 200:
        return response.json()
    else:
        print('Error:', response.status_code)
        raise HTTPException(status_code=response.status_code, detail=response.json())

@router.get("/find-recommendations", response_model=list)
def find_anime(
    anime_id: int,
    settings: Annotated[Settings, Depends(get_settings)]
):
    # Initialize constants
    NUM_OF_GOOD_RECOMMENDATIONS = 8
    NUM_OF_BAD_RECOMMENDATIONS = 2
    start_time = time.time()

    # Get the selected anime
    try: 
        response = requests.get(
            f'https://api.myanimelist.net/v2/anime/{anime_id}?fields=id,title,main_picture,synopsis,related_anime', 
            headers={'X-MAL-Client-ID': settings.client_id}
        )
        response.headers['Content-Type'] = 'application/json'
        selectedAnime = response.json()
    except requests.RequestException as e:
        print(f'Error fetching selected anime: {e}')
        raise HTTPException(status_code=500, detail='Error fetching selected anime')

    # Process the related titles to the anime
    related_titles = find_related_titles(selectedAnime, settings)
    
    # Fetch the ranking data from the API
    try:
        response = requests.get(
            f'https://api.myanimelist.net/v2/anime/ranking?ranking_type=all&limit=500&fields=id,rank,title,main_picture,synopsis', 
            headers={'X-MAL-Client-ID': settings.client_id}
        )
        response.headers['Content-Type'] = 'application/json'
        rank_data = response.json()['data']
    except requests.RequestException as e:
        print(f'Error fetching ranking data: {e}')
        raise HTTPException(status_code=500, detail='Error fetching ranking data')

    # Calculate the cosine similarity between the selected anime and the ranking data
    doc = nlp(selectedAnime['synopsis']).vector.reshape(1, -1)
    vectors = [nlp(anime['node']['synopsis']).vector for anime in rank_data]
    anime_similarities = cosine_similarity(doc, vectors)[0]
    recommended = find_recommendations(anime_similarities, NUM_OF_GOOD_RECOMMENDATIONS, NUM_OF_BAD_RECOMMENDATIONS, rank_data, related_titles)
    for i in recommended:
        print(rank_data[i]['node']['title'])

    runtime = time.time() - start_time
    print(f"Execution time: {runtime} seconds")
    
    if response.status_code == 200:
        return response.json().get('data', [])
    else:
        print('Error:', response.status_code)
        raise HTTPException(status_code=response.status_code, detail=response.json()
)

