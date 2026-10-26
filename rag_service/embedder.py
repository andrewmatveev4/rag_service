import os

import openai
from dotenv import load_dotenv
from openai import OpenAI

from rag_service.errors import ConfigError, LLMAPIError, LLMOutputError
from rag_service.retry import with_retry

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
EMBED_BATCH_SIZE = 100


def get_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ConfigError("OPENAI_API_KEY не задан: заполни .env")
    return OpenAI(api_key=api_key, max_retries=0, timeout=30)


def _embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    try:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    except openai.RateLimitError as e:
        if e.code == "insufficient_quota":
            raise LLMAPIError("Закончился баланс API", retryable=False) from e
        raise LLMAPIError(f"Слишком много запросов: {e}", retryable=True) from e
    except (openai.APIConnectionError, openai.InternalServerError) as e:
        raise LLMAPIError(f"Временная ошибка API: {e}", retryable=True) from e
    except openai.APIError as e:
        raise LLMAPIError(f"Ошибка API: {e}", retryable=False) from e
    vectors = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
    if len(vectors) != len(texts):
        raise LLMOutputError(f"Ждали {len(texts)} векторов, пришло {len(vectors)}", "")
    return vectors


def embed_texts(texts: list[str], client: OpenAI | None = None) -> list[list[float]]:
    if not texts:
        return []
    client = client or get_client()
    vectors = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        vectors.extend(with_retry(lambda: _embed_batch(client, batch)))
    return vectors
