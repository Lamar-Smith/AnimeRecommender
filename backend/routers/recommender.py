import requests
import spacy
import json
from sklearn.metrics.pairwise import cosine_similarity
from settings import Settings, get_settings
from fastapi import APIRouter, HTTPException, Depends
from typing import Annotated

router = APIRouter()

NUM_OF_GOOD_RECOMMENDATIONS = 8
NUM_OF_BAD_RECOMMENDATIONS = 3

nlp = spacy.load('en_core_web_lg')

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
    response = requests.get(
        f'https://api.myanimelist.net/v2/anime/{anime_id}?fields=id,title,main_picture,synopsis', 
        headers={'X-MAL-Client-ID': settings.client_id}
    )
    response.headers['Content-Type'] = 'application/json'

    selectedAnime = response.json()
    doc = [nlp(selectedAnime['synopsis']).vector]

    response = requests.get(
        f'https://api.myanimelist.net/v2/anime/ranking?ranking_type=all&limit=500&fields=id,rank,title,main_picture,synopsis', 
        headers={'X-MAL-Client-ID': settings.client_id}
    )
    response.headers['Content-Type'] = 'application/json'
    rank_data = response.json()['data']
    print(rank_data)

    print(nlp(rank_data[0]['node']['synopsis']))

    vectors = [nlp(anime['node']['synopsis']).vector for anime in rank_data]
    anime_similarities = cosine_similarity(doc, vectors)

    print(rank_data)
    
    if response.status_code == 200:
        return response.json().get('data', [])
    else:
        print('Error:', response.status_code)
        raise HTTPException(status_code=response.status_code, detail=response.json()
)

