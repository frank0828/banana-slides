"""
Text generation using AiHubMix standard HTTP API (OpenAI compatible format)
直接调用 /v1/chat/completions 接口，避免 genai SDK 挂死问题

Supports both API-key mode and compatible proxy (e.g., AiHubMix)
"""
import logging
import httpx
from typing import Generator
from urllib.parse import urlparse
from .base import TextProvider

logger = logging.getLogger(__name__)


class GenAITextProvider(TextProvider):
    """Text generation using OpenAI-compatible HTTP API
    
    Uses direct HTTP calls instead of GenAI SDK to avoid hanging issues
    with proxy environments like AiHubMix.
    """

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        api_key: str = None,
        api_base: str = None,
        vertexai: bool = False,  # 保留接口兼容，但 HTTP 模式不支持
        project_id: str = None,
        location: str = None,
    ):
        if vertexai:
            logger.warning("Vertex AI mode not supported in HTTP API mode, using API key mode")
        
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

        self._client = self._create_client()
        logger.info(f"[TextProvider] Using HTTP API: {self.chat_url}, model: {self.model}")

    def _create_client(self, force_refresh: bool = False) -> httpx.Client:
        """创建 httpx 客户端（自动探测直连/系统代理）"""
        from utils.net_utils import get_ai_proxy
        return httpx.Client(
            verify=False,
            trust_env=False,
            proxy=get_ai_proxy(force_refresh=force_refresh),
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    def _post_with_retry(self, payload: dict, headers: dict) -> httpx.Response:
        """POST 请求；连接类错误时强制重新探测网络并重试（网络环境可能已切换）"""
        last_exc = None
        for attempt in range(2):
            if attempt > 0:
                logger.warning(f"[TextProvider] Retrying after error: {last_exc}")
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = self._create_client(force_refresh=True)
            try:
                response = self._client.post(self.chat_url, json=payload, headers=headers)
                response.raise_for_status()
                return response
            except Exception as e:
                err_str = str(e)
                if any(kw in err_str for kw in ('Timeout', 'Connection', 'Remote', 'Read', 'timed out')):
                    last_exc = e
                    if attempt == 1:
                        raise
                    continue
                raise
        raise Exception(f'Request failed: {last_exc}')

    def generate_text(self, prompt: str, thinking_budget: int = 0) -> str:
        """
        Generate text via OpenAI-compatible API

        Args:
            prompt: The input prompt
            thinking_budget: Not used in HTTP mode (kept for interface compatibility)
            
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

        response = self._post_with_retry(payload, headers)
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    def generate_with_image(self, prompt: str, image_path: str, thinking_budget: int = 0) -> str:
        """
        Generate text with image input using vision API
        
        Args:
            prompt: The input prompt
            image_path: Path to the image file
            thinking_budget: Not used in HTTP mode
            
        Returns:
            Generated text
        """
        import base64
        from PIL import Image
        import io
        
        # 加载并编码图片
        img = Image.open(image_path)
        buffer = io.BytesIO()
        fmt = img.format or 'PNG'
        img.save(buffer, format=fmt)
        b64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        mime_type = f"image/{fmt.lower()}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}}
                ]
            }],
            "max_tokens": 4096,
        }

        response = self._post_with_retry(payload, headers)
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def generate_text_stream(self, prompt: str, thinking_budget: int = 0) -> Generator[str, None, None]:
        """
        Stream text using OpenAI-compatible streaming API
        
        Args:
            prompt: The input prompt
            thinking_budget: Not used in HTTP mode
            
        Yields:
            Text chunks as they arrive
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "stream": True,
        }

        with self._client.stream("POST", self.chat_url, json=payload, headers=headers) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        import json
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except Exception:
                        continue
