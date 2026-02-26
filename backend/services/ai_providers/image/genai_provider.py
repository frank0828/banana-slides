"""
Google GenAI SDK implementation for image generation
"""
import logging
import ssl as _ssl
import urllib.request as _urllib_req
import httpx as _httpx
from typing import Optional, List
from google import genai
from google.genai import types
from PIL import Image
from .base import ImageProvider

logger = logging.getLogger(__name__)

# 处理本地代理（Clash/V2Ray 等）与 genai SDK 的 SSL 兼容性问题
# 背景：通过本地代理（如 127.0.0.1:20809）时，代理做 MITM 替换证书，但 Python 不信任代理 CA → SSL 握手超时
# 解决方案：直接给 genai SDK 注入自定义 httpx.Client（携带代理 + 跳过 SSL 验证）
# 注意：HttpOptions.timeout 单位是毫秒（SDK 内部会 /1000 换算为秒）

def _get_local_proxy() -> str | None:
    """检测 Windows 注册表中的系统代理，只返回本机代理地址（127.x/localhost）"""
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

def _build_httpx_client() -> _httpx.Client:
    """构建适配本机代理环境的 httpx.Client（不走 trust_env，避免注册表代理干扰）
    
    注意：max_keepalive_connections=0 禁用连接池，避免"Server disconnected"错误。
    genai 调用耗时长（30s+），keep-alive 连接很容易被代理/服务端关闭，
    禁用后每次请求都新建连接，牺牲极少性能换取稳定性。
    """
    kwargs: dict = {
        'follow_redirects': True,
        'verify': False,   # 本地代理 MITM / 直连证书链问题均跳过验证
        'trust_env': False,  # 不自动读取系统代理，由我们手动控制
        # 禁用连接池：避免长耗时请求后连接被对端关闭导致 RemoteProtocolError
        'limits': _httpx.Limits(
            max_keepalive_connections=0,
            max_connections=10,
        ),
        # 直接在 httpx.Client 上设置超时，确保不依赖 SDK 内部透传
        # connect=30s：建立连接超时；read=360s：等待响应超时（图像生成耗时长）
        'timeout': _httpx.Timeout(360.0, connect=30.0),
    }
    if _SYSTEM_PROXY:
        kwargs['proxy'] = _SYSTEM_PROXY
    return _httpx.Client(**kwargs)


class GenAIImageProvider(ImageProvider):
    """Image generation using Google GenAI SDK"""
    
    def __init__(self, api_key: str, api_base: str = None, model: str = "gemini-3-pro-image-preview"):
        """
        Initialize GenAI image provider
        
        Args:
            api_key: Google API key
            api_base: API base URL (for proxies like aihubmix)
            model: Model name to use
        """
        # HttpOptions.timeout 单位是毫秒（SDK 内部会除以 1000 转换为秒）
        # 300_000ms = 300s = 5 分钟，给图像生成足够的时间
        # httpx_client：直接传入自定义客户端，绕过 SDK 内部 SSL context 创建逻辑
        _custom_client = _build_httpx_client()
        _http_opts = types.HttpOptions(
            base_url=api_base if api_base else None,
            timeout=300_000,
            httpx_client=_custom_client,
        )
        self.client = genai.Client(
            http_options=_http_opts,
            api_key=api_key
        )
        self.model = model
        # 保存用于重试时重建 client
        self._api_key = api_key
        self._api_base = api_base
    
    def generate_image(
        self,
        prompt: str,
        ref_images: Optional[List[Image.Image]] = None,
        aspect_ratio: str = "16:9",
        resolution: str = "2K"
    ) -> Optional[Image.Image]:
        """
        Generate image using Google GenAI SDK
        
        Args:
            prompt: The image generation prompt
            ref_images: Optional list of reference images
            aspect_ratio: Image aspect ratio
            resolution: Image resolution (supports "1K", "2K", "4K")
            
        Returns:
            Generated PIL Image object, or None if failed
        """
        try:
            # Build contents list with prompt and reference images
            contents = []
            
            # Add reference images first (if any)
            if ref_images:
                for ref_img in ref_images:
                    contents.append(ref_img)
            
            # Add text prompt
            contents.append(prompt)
            
            logger.debug(f"Calling GenAI API for image generation with {len(ref_images) if ref_images else 0} reference images...")
            logger.debug(f"Config - aspect_ratio: {aspect_ratio}, resolution: {resolution}")
            
            # 重试机制：RemoteProtocolError（连接被关闭）时自动重建连接重试
            import time as _time
            last_exc = None
            for attempt in range(3):
                if attempt > 0:
                    logger.warning(f"Retrying GenAI API call (attempt {attempt + 1}/3) after error: {last_exc}")
                    _time.sleep(2 * attempt)  # 递增等待：2s, 4s
                    # 重建 httpx client 确保使用新连接（使用保存的 api_key / api_base）
                    _new_client = _build_httpx_client()
                    _http_opts = types.HttpOptions(
                        base_url=self._api_base if self._api_base else None,
                        timeout=300_000,
                        httpx_client=_new_client,
                    )
                    self.client = genai.Client(
                        http_options=_http_opts,
                        api_key=self._api_key,
                    )
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_modalities=['TEXT', 'IMAGE'],
                            image_config=types.ImageConfig(
                                aspect_ratio=aspect_ratio,
                            ),
                        )
                    )
                    break  # 成功则跳出循环
                except Exception as retry_exc:
                    err_str = str(retry_exc)
                    # 仅对连接断开类错误重试（超时不重试，重试只会等更久）
                    if any(kw in err_str for kw in ('RemoteProtocolError', 'Server disconnected', 'ConnectionError', 'RemoteDisconnected')):
                        last_exc = retry_exc
                        if attempt == 2:
                            raise  # 最后一次失败则抛出
                        continue
                    raise  # 非连接错误直接抛出
            
            logger.debug("GenAI API call completed")
            
            # Extract image from response
            for i, part in enumerate(response.parts):
                if part.text is not None:
                    logger.debug(f"Part {i}: TEXT - {part.text[:100] if len(part.text) > 100 else part.text}")
                else:
                    try:
                        logger.debug(f"Part {i}: Attempting to extract image...")
                        image = part.as_image()
                        if image:
                            logger.debug(f"Successfully extracted image from part {i}")
                            return image
                    except Exception as e:
                        logger.debug(f"Part {i}: Failed to extract image - {str(e)}")
            
            # No image found in response
            error_msg = "No image found in API response. "
            if response.parts:
                error_msg += f"Response had {len(response.parts)} parts but none contained valid images."
            else:
                error_msg += "Response had no parts."
            
            raise ValueError(error_msg)
            
        except Exception as e:
            error_detail = f"Error generating image with GenAI: {type(e).__name__}: {str(e)}"
            logger.error(error_detail, exc_info=True)
            raise Exception(error_detail) from e

