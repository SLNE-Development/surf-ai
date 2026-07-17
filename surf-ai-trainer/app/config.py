from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SURF_AI_TRAINER_")

    postgres_dsn: str = "postgresql://surf_ai:surf_ai@localhost:5432/surf_ai"
    s3_endpoint: str = "http://localhost:9000"
    s3_bucket: str = "surf-ai"
    s3_access_key: str = "surfai"
    s3_secret_key: str = "surfaikey"
    embedding_model_id: str = "intfloat/multilingual-e5-small"
    embedding_prefix: str = "query: "
    base_head_checkpoint_s3_key: str = "models/head/base-checkpoint.pt"
