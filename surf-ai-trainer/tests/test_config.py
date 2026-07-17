from app.config import Settings


def test_settings_loads_dev_defaults():
    settings = Settings()
    assert settings.embedding_prefix == "query: "
    assert settings.embedding_model_id == "intfloat/multilingual-e5-small"
    assert settings.s3_bucket == "surf-ai"
