"""
Image generation using Gemini REST API (HTTP)
直接调用 Gemini REST API，避免 GenAI SDK 在代理环境下挂死

Supports both API-key mode and compatible proxy (e.g., AiHubMix)
"""
import logging
import base64
import io
import time
import httpx
from typing import Optional, List
from PIL import Image
from .base import ImageProvider

logger = logging.getLogger(__name__)


class GenAIImageProvider(ImageProvider):
    """Image generation using Gemini REST API (HTTP)
    
    Uses direct HTTP calls instead of GenAI SDK to avoid hanging issues
    with proxy environments like AiHubMix.
    """
    
    def __init__(
        self,
        model: str = "gemini-3-pro-image-preview",
        api_key: str = None,
        api_base: str = None,
        vertexai: bool = False,  # 保留接口兼容，但 HTTP 模式不支持
        project_id: str = None,
        location: str = None,
    ):
        """
        Initialize Gemini image provider
        
        Args:
            model: Model name to use
            api_key: API key
            api_base: API base URL (e.g., https://aihubmix.com/gemini)
            vertexai: Not supported in HTTP mode (kept for interface compatibility)
            project_id: Not supported in HTTP mode
            location: Not supported in HTTP mode
        """
        if vertexai:
            logger.warning("Vertex AI mode not supported in HTTP API mode, using API key mode")
        
        self.api_key = api_key
        self.model = model
        
        # 构建 API URL
        # 使用流式端点（streamGenerateContent + SSE）：
        # 非流式端点在整张图生成完毕前不返回任何字节，长时间静默会被
        # 网关/代理的 60 秒空闲超时掐断；流式端点会持续吐数据，规避该问题。
        base = api_base or "https://generativelanguage.googleapis.com"
        self.api_url = f"{base}/v1beta/models/{model}:streamGenerateContent?alt=sse"

        # httpx 客户端：读超时 300 秒（4K 大图生成耗时长），禁用 SSL 验证（代理环境）
        self._timeout = httpx.Timeout(300.0, connect=30.0)
        self._limits = httpx.Limits(max_keepalive_connections=0, max_connections=10)
        self._client = self._create_client()

        logger.info(f"[ImageProvider] Using streaming HTTP API: {self.api_url}")
    
    def _create_client(self, force_refresh: bool = False, force_direct: bool = False) -> httpx.Client:
        """创建新的 httpx 客户端（自动探测直连/系统代理）

        Args:
            force_refresh: 强制重新探测网络（网络环境可能已切换）
            force_direct: 强制直连，忽略代理（代理对长连接断流时的备选路线）
        """
        from utils.net_utils import get_ai_proxy
        proxy = None if force_direct else get_ai_proxy(force_refresh=force_refresh)
        logger.info(f"[ImageProvider] 客户端路线: {'直连' if not proxy else proxy}")
        return httpx.Client(
            verify=False,
            trust_env=False,
            proxy=proxy,
            timeout=self._timeout,
            limits=self._limits,
        )
    
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
        resolution: str = "2K",
        enable_thinking: bool = True,  # 保留接口兼容，HTTP 模式暂不支持
        thinking_budget: int = 1024
    ) -> Optional[Image.Image]:
        """
        Generate image using Gemini REST API
        
        Args:
            prompt: The image generation prompt
            ref_images: Optional list of reference images
            aspect_ratio: Image aspect ratio (16:9, 1:1, 9:16, 3:4, 4:3)
            resolution: Image resolution (1K, 2K, 4K)
            enable_thinking: Not supported in HTTP mode (kept for interface compatibility)
            thinking_budget: Not supported in HTTP mode
            
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
            # image_config: aspect_ratio + image_size
            payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"],
                    "image_config": {
                        "aspect_ratio": aspect_ratio,
                        "image_size": resolution
                    }
                }
            }
            
            logger.info(f"[ImageProvider] Generating image: aspect_ratio={aspect_ratio}, image_size={resolution}")
            
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            }
            
            logger.debug(f"Calling Gemini API with {len(ref_images) if ref_images else 0} reference images...")

            # 重试机制：网络抖动时重建客户端（重新探测直连/代理）后重试
            last_exc = None
            max_attempts = 3
            image_b64 = None
            finish_reason = ""
            for attempt in range(max_attempts):
                if attempt > 0:
                    logger.warning(f"Retrying API call (attempt {attempt + 1}/{max_attempts}) after error: {last_exc}")
                    time.sleep(min(3 * attempt, 6))
                    try:
                        self._client.close()
                    except Exception:
                        pass
                    self._client = self._create_client(force_refresh=True)

                try:
                    image_b64, finish_reason = self._stream_request(payload, headers)
                    break
                except (httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError):
                    # 不重试：请求体已完整送达，服务端正在/已经出图（即已计费），
                    # 重试只会再生成一张、再扣一次费，而调用方最多拿到一张。
                    raise
                except httpx.TransportError as e:
                    # 仅重试"请求未完整送达"的传输错误（连接失败、代理错误、写超时等）。
                    # 按异常类型判断，不要用字符串匹配：ReadTimeout 的消息是
                    # "The read operation timed out"，大小写关键词匹配不上。
                    last_exc = e
                    if attempt == max_attempts - 1:
                        raise
                    continue

            logger.debug("API call completed")

            if image_b64:
                logger.debug("Successfully extracted image from stream response")
                return self._base64_to_image(image_b64)

            # 检查是否被内容策略拒绝
            if finish_reason and finish_reason != "STOP":
                raise ValueError(f"Generation blocked: {finish_reason}")

            raise ValueError("No image found in API response")
            
        except Exception as e:
            # 链路超时/断连是这条通路最常见的失败，给出可操作的中文提示，
            # 而不是把 httpx 堆栈原样抛给用户
            if isinstance(e, (httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.ConnectTimeout)):
                error_detail = (
                    f"图片生成失败（网络链路超时：{type(e).__name__}）。"
                    f"该 API 在整图生成完毕前不返回任何数据，若生成耗时超过链路空闲超时（约 60 秒）即被断开。"
                    f"建议：降低分辨率至 2K 或 1K、减少参考图数量，或调大本地代理的空闲超时设置。"
                )
                logger.error(error_detail)
                raise Exception(error_detail) from e

            error_detail = f"Error generating image: {type(e).__name__}: {str(e)}"
            logger.error(error_detail, exc_info=True)
            raise Exception(error_detail) from e

    def _stream_request(self, payload: dict, headers: dict) -> tuple[Optional[str], str]:
        """通过 SSE 流式接口发起请求，返回 (图片 base64, finishReason)

        流式端点会持续返回数据块，避免非流式请求在长时间静默后
        被网关/代理的空闲超时（约 60 秒）断开连接。
        """
        import json as _json

        image_parts: List[str] = []
        finish_reason = ""

        with self._client.stream('POST', self.api_url, json=payload, headers=headers) as response:
            if response.status_code != 200:
                body = response.read()[:1000]
                raise ValueError(f"API error {response.status_code}: {body!r}")

            for line in response.iter_lines():
                if not line or not line.startswith('data: '):
                    continue
                chunk = line[6:].strip()
                if chunk == '[DONE]':
                    break
                try:
                    data = _json.loads(chunk)
                except Exception:
                    continue

                for candidate in data.get('candidates', []):
                    reason = candidate.get('finishReason', '')
                    if reason:
                        finish_reason = reason
                    for part in candidate.get('content', {}).get('parts', []):
                        if 'inlineData' in part:
                            b64 = part['inlineData'].get('data', '')
                            if b64:
                                image_parts.append(b64)

        return (''.join(image_parts) if image_parts else None), finish_reason
