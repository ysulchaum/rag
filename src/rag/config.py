import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from openai import OpenAI

load_dotenv()


@dataclass(frozen=True)
class Settings:
	database_url: str = os.getenv(
		"DATABASE_URL",
		"postgresql://rag_user:rag_password@localhost:5450/rag_db",
	)
	deepseek_api_key: str | None = os.getenv("DEEPSEEK_API_KEY")
	deepseek_base_url: str = os.getenv(
		"DEEPSEEK_BASE_URL", "https://api.deepseek.com"
	)
	deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
	embedding_base_url: str = os.getenv(
		"EMBEDDING_BASE_URL", "http://localhost:11434"
	)
	embedding_model: str = os.getenv(
		"EMBEDDING_MODEL", "bge-m3"
	)


settings = Settings()


@dataclass(frozen=True)
class TokenUsage:
	input_tokens: int | None
	output_tokens: int | None
	total_tokens: int | None


class DeepSeekClient:
	def __init__(self, configuration: Settings) -> None:
		self.configuration = configuration

	def generate(self, prompt: str) -> str:
		return self.generate_with_usage(prompt)[0]

	def generate_with_usage(self, prompt: str) -> tuple[str, TokenUsage]:
		if not self.configuration.deepseek_api_key:
			raise RuntimeError("Set DEEPSEEK_API_KEY to use DeepSeek generation.")

		client = OpenAI(
			api_key=self.configuration.deepseek_api_key,
			base_url=self.configuration.deepseek_base_url,
		)
		response = client.chat.completions.create(
			model=self.configuration.deepseek_model,
			messages=[{"role": "user", "content": prompt}],
		)
		content = response.choices[0].message.content
		if content is None:
			raise RuntimeError("DeepSeek returned an empty response.")
		usage = response.usage
		return content, TokenUsage(
			input_tokens=usage.prompt_tokens if usage else None,
			output_tokens=usage.completion_tokens if usage else None,
			total_tokens=usage.total_tokens if usage else None,
		)


embedding_client = OllamaEmbeddings(
	model=settings.embedding_model,
	base_url=settings.embedding_base_url,
)


def embed_text(text: str) -> list[float]:
	return embedding_client.embed_query(text)


llm_client = DeepSeekClient(settings)
