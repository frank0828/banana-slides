"""
OpenAI SDK implementation for text generation
"""
import logging
import urllib.request as _urllib_req
import httpx as _httpx
from openai import OpenAI
from .base import TextProvider
from config import get_config

logger = logging.getLogger(__name__)


def _get_local_proxy() -> str | None:
    try:
        proxies = _urllib_req.getproxies_registry()
        for scheme in ('https', 'http'):
            addr = proxies.get(scheme, '')
            if addr and ('127.' in addr or 'localhost' in addr):
                return addr
    except Exception:
        pass
    return None


_SYSTEM_PROXY = _get_local_proxy()


def _build_http_client() -> _httpx.Client:
    kwargs: dict = {
        'follow_redirects': True,
        'verify': False,
        'trust_env': False,
        'timeout': _httpx.Timeout(300.0, connect=30.0),
    }
    if _SYSTEM_PROXY:
        kwargs['proxy'] = _SYSTEM_PROXY
    return _httpx.Client(**kwargs)


class OpenAITextProvider(TextProvider):
    """Text generation using OpenAI SDK (compatible with Gemini via proxy)"""
    
    def __init__(self, api_key: str, api_base: str = None, model: str = "gemini-3-flash-preview"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=get_config().OPENAI_TIMEOUT,
            max_retries=0,
            http_client=_build_http_client(),
        )
        self.model = model
    
    def generate_text(self, prompt: str, thinking_budget: int = 1000) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
