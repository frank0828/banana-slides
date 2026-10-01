"""
网络工具：自动探测 AI 服务（aihubmix）的可达方式

背景：
- 有些网络环境下直连 aihubmix.com 可用，本地代理反而断连（ProxyError）
- 有些网络环境下直连超时，必须走系统代理（如 V2Ray 127.0.0.1:10809）
本模块统一探测一次并缓存结果，供 httpx / requests 客户端复用。
"""
import logging
import threading
import time

logger = logging.getLogger(__name__)

_PROBE_URL = 'https://aihubmix.com'
_PROBE_TTL = 300  # 探测结果有效期 5 分钟，过期自动重新探测（网络环境可能随时切换）
_lock = threading.Lock()
_cached_proxy = None
_probed = False
_probed_at = 0.0


def _get_system_proxy():
    """获取系统代理地址（环境变量或 Windows 注册表）"""
    try:
        import urllib.request
        proxies = urllib.request.getproxies()
        proxy = proxies.get('https') or proxies.get('http')
        if proxy and not proxy.lower().startswith('http'):
            proxy = 'http://' + proxy
        return proxy
    except Exception:
        return None


def get_ai_proxy(force_refresh: bool = False):
    """
    返回访问 AI 服务应使用的代理 URL；直连可用时返回 None。
    结果进程内缓存，force_refresh=True 时重新探测。
    """
    global _cached_proxy, _probed, _probed_at
    with _lock:
        cache_fresh = _probed and (time.time() - _probed_at) < _PROBE_TTL
        if cache_fresh and not force_refresh:
            return _cached_proxy

        import httpx
        probe_timeout = httpx.Timeout(8.0, connect=5.0)

        # 1. 先试直连
        try:
            with httpx.Client(verify=False, trust_env=False, timeout=probe_timeout) as c:
                c.get(_PROBE_URL)
            _cached_proxy = None
            _probed = True
            _probed_at = time.time()
            logger.info('[NetProbe] 直连 aihubmix 可用，不使用代理')
            return None
        except Exception as e:
            logger.info(f'[NetProbe] 直连失败（{type(e).__name__}），尝试系统代理...')

        # 2. 直连不通，试系统代理
        proxy = _get_system_proxy()
        if proxy:
            try:
                with httpx.Client(verify=False, proxy=proxy, timeout=probe_timeout) as c:
                    c.get(_PROBE_URL)
                _cached_proxy = proxy
                _probed = True
                _probed_at = time.time()
                logger.info(f'[NetProbe] 使用系统代理: {proxy}')
                return proxy
            except Exception as e:
                logger.warning(f'[NetProbe] 系统代理 {proxy} 也不可用（{type(e).__name__}）')

        # 3. 都不通，默认直连（让上层报出真实错误）
        _cached_proxy = None
        _probed = True
        _probed_at = time.time()
        logger.warning('[NetProbe] 直连与代理均不可用，默认直连')
        return None


def get_requests_proxies(force_refresh: bool = False):
    """返回 requests 库格式的 proxies 字典"""
    proxy = get_ai_proxy(force_refresh)
    if proxy:
        return {'http': proxy, 'https': proxy}
    return {'http': None, 'https': None}
