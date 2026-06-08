from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Template Service"
    app_env: str = "development"
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "app"
    db_user: str = "postgres"
    db_pass: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
