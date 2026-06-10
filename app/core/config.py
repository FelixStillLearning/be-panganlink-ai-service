from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Template Service"
    app_env: str = "development"
    db_host: str = "mysql"
    db_port: int = 3306
    db_name: str = "panganlink_db"
    db_user: str = "mysqluser"
    db_pass: str = "password"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
