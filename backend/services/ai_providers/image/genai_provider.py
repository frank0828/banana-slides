"""
Image generation using Gemini REST API (HTTP)
直接调用 Gemini REST API，避免 GenAI SDK 在代理环境下挂死
"""
import logging
import base64
import io
import time
import httpx
from typing import Optional, List
from urllib.parse import urlparse
from PIL import Image
from .base import ImageProvider

logger = logging.getLogger(__name__)


class GenAIImageProvider(ImageProvider):
    """Image generation using Gemini REST API (HTTP)"""
    
    def __init__(self, api_key: str, api_base: str = None, model: str = "gemini-2.0-flash-exp-image-generation"):
        """
        Initialize Gemini image provider
        
        Args:
            api_key: API key
            api_base: API base URL (e.g., https://aihubmix.com/gemini)
            model: Model name to use
        """
        self.api_key = api_key
        self.model = model
        
        # 构建 API URL
        if api_base:
            # https://aihubmix.com/gemini -> https://aihubmix.com/gemini/v1beta/models/{model}:generateContent
            self.api_url = f"{api_base}/v1beta/models/{model}:generateContent"
        else:
            # 默认 Google 官方 API
            self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        
        # httpx 客户端：180秒超时，禁用 SSL 验证（代理环境）
        self._timeout = httpx.Timeout(180.0, connect=30.0)
        self._limits = httpx.Limits(max_keepalive_connections=0, max_connections=10)
        self._client = self._create_client()
    
    def _create_client(self) -> httpx.Client:
        """创建新的 httpx 客户端"""
        return httpx.Client(
            verify=False,
            trust_env=False,
            timeout=self._timeout,
            limits=self._limits,
        )
        logger.info(f"[ImageProvider] Using HTTP API: {self.api_url}")
    
    def _image_to_base64(self, img: Image.Image) -> tuple[str, str]:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        fmt = img.format or 'PNG'
        img.save(buffer, format=fmt)
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        mime = f"image/{fmt.lower()}"
        return b64, mime
    
    def _base64_to_image(self, b64_data: str) -> Image.Image:
        """Convert base64 string to PIL Image"""
        img_bytes = base64.b64decode(b64_data)
        return Image.open(io.BytesIO(img_bytes))
    
    def generate_image(
        self,
        prompt: str,
        ref_images: Optional[List[Image.Image]] = None,
        aspect_ratio: str = "16:9",
        resolution: str = "2K"
    ) -> Optional[Image.Image]:
        """
        Generate image using Gemini REST API
        
        Args:
            prompt: The image generation prompt
            ref_images: Optional list of reference images
            aspect_ratio: Image aspect ratio
            resolution: Image resolution (supports "1K", "2K", "4K")
            
        Returns:
            Generated PIL Image object, or None if failed
        """
        try:
            # 构建 contents
            parts = []
            
            # 添加参考图片
            if ref_images:
                for ref_img in ref_images:
                    b64, mime = self._image_to_base64(ref_img)
                    parts.append({
                        "inline_data": {
                            "mime_type": mime,
                            "data": b64
                        }
                    })
            
            # 添加文本 prompt
            parts.append({"text": prompt})
            
            # 构建请求体
            payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"]
                }
            }
            
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            }
            
            logger.debug(f"Calling Gemini API with {len(ref_images) if ref_images else 0} reference images...")
            logger.debug(f"Config - aspect_ratio: {aspect_ratio}, resolution: {resolution}")
            
            # 重试机制（超时/连接错误时重建客户端重试）
            last_exc = None
            for attempt in range(3):
                if attempt > 0:
                    logger.warning(f"Retrying API call (attempt {attempt + 1}/3) after error: {last_exc}")
                    time.sleep(3 * attempt)  # 3s, 6s
                    # 重建客户端确保新连接
                    try:
                        self._client.close()
                    except Exception:
                        pass
                    self._client = self._create_client()
                
                try:
                    response = self._client.post(
                        self.api_url,
                        json=payload,
                        headers=headers
                    )
                    response.raise_for_status()
                    data = response.json()
                    break
                except Exception as e:
                    err_str = str(e)
                    if any(kw in err_str for kw in ('Timeout', 'Connection', 'Remote', 'Read')):
                        last_exc = e
                        if attempt == 2:
                            raise
                        continue
                    raise
            
            logger.debug("API call completed")
            
            # 解析响应中的图片
            candidates = data.get("candidates", [])
            for candidate in candidates:
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                for part in parts:
                    if "inlineData" in part:
                        inline = part["inlineData"]
                        b64_data = inline.get("data", "")
                        if b64_data:
                            logger.debug("Successfully extracted image from response")
                            return self._base64_to_image(b64_data)
            
            # 检查是否被内容策略拒绝
            if candidates:
                finish_reason = candidates[0].get("finishReason", "")
                if finish_reason and finish_reason != "STOP":
                    raise ValueError(f"Generation blocked: {finish_reason}")
            
            raise ValueError("No image found in API response")
            
        except Exception as e:
            error_detail = f"Error generating image: {type(e).__name__}: {str(e)}"
            logger.error(error_detail, exc_info=True)
            raise Exception(error_detail) from e
