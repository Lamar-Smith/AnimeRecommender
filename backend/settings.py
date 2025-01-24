from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    client_id: str
    
    class Config:
        env_file = ".env"

settings = Settings()

def get_settings():
    return Settings()