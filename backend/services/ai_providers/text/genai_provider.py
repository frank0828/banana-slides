"""
Text generation using AiHubMix standard HTTP API (OpenAI compatible format)
直接调用 /v1/chat/completions 接口，避免 genai SDK 挂死问题
"""
import logging
import httpx
from urllib.parse import urlparse
from .base import TextProvider

logger = logging.getLogger(__name__)


class GenAITextProvider(TextProvider):
    """Text generation using OpenAI-compatible HTTP API"""

    def __init__(self, api_key: str, api_base: str = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model

        # 从 api_base 推导标准接口地址
        # 例： https://aihubmix.com/gemini → https://aihubmix.com/v1/chat/completions
        if api_base:
            parsed = urlparse(api_base)
            self.chat_url = f"{parsed.scheme}://{parsed.netloc}/v1/chat/completions"
        else:
            # 未配置 api_base 时回退到 Google 官方接口
            self.chat_url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

        self._client = httpx.Client(
            verify=False,
            trust_env=False,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        logger.info(f"[TextProvider] Using HTTP API: {self.chat_url}, model: {self.model}")

    def generate_text(self, prompt: str, thinking_budget: int = 1000) -> str:
        """
        Generate text via OpenAI-compatible API

        Args:
            prompt: The input prompt
            thinking_budget: Unused in HTTP mode, kept for interface compatibility

        Returns:
            Generated text string
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
        }

        response = self._client.post(self.chat_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
