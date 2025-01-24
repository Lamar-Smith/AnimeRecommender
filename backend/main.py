from fastapi import FastAPI
from routers.recommender import router as recommender_router

app = FastAPI()


@app.get("/")
def root():
    return {"Hello": "World"}

app.include_router(recommender_router)