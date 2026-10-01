"""
image2 —— 独立的 AiHubMix GPT Image 出图程序（gpt-image-2 / 2.5，不依赖后端）

用法：
    # 文生图（无参考图，默认 gpt-image-2）
    uv run python image2/gen.py --prompt "一只赛博朋克风格的猫" --size 2K --ratio 16:9

    # 指定模型：gpt-image-2.5-flare（快速高质量）/ gpt-image-2.5-sunburst（高精度编辑）
    uv run python image2/gen.py --model gpt-image-2.5-flare --prompt "..." --quality xhigh

    # 图生图 / 多图融合（带参考图，走 /images/edits）
    uv run python image2/gen.py --prompt "把这张图修复成高清唐卡" --ref a.png b.png --size 4K --ratio 9:16

    # 直接指定像素 size（覆盖 --size/--ratio）
    uv run python image2/gen.py --prompt "..." --pixel-size 2160x3840

    # 透明背景（官方预览功能，不额外收费；alpha 需 png/webp 承载）
    uv run python image2/gen.py --prompt "一个图标，只要主体" --background transparent

可选模型见 MODEL_SPECS。

协议路线（2026-09-22 起默认异步）：
    - 默认走 AiHubMix 原生异步协议：POST /ai/v1/images/generations（async:true）
      创建任务 → GET /ai/v1/tasks/{id} 每 15s 轮询 → GET /ai/v1/tasks/{id}/content 下载。
      实测 max 画质 2160x3840 约 76s 完成；创建/轮询/下载都是短请求，
      彻底绕开同步链路「约 60s 空闲即被代理/网关掐断」的问题。
      注意：产物仅保留 2 小时（过期 410）；必须用创建时同一把 Key 查询/下载。
    - --sync 回退旧同步 /v1/images/* 路径：单个长连接等到出图为止，
      超过 ~60s 的生成（high/xhigh/max、大尺寸）大概率被链路掐断，仅作备用。

配置来源（优先级从高到低）：
    1) 命令行 --api-key / --api-base / --proxy
    2) 环境变量 AIHUBMIX_API_KEY / AIHUBMIX_API_BASE / AIHUBMIX_PROXY
    3) 项目 backend/instance/database.db 的 settings 表
    4) 系统代理（自动探测）

本程序固化了大量实测结论，详见 README.md。
"""
import argparse
import base64
import io
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

import requests
from PIL import Image

try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

# ----------------------------------------------------------------------------
# 常量（均来自实测，勿随意改动，改动前先读 README.md）
# ----------------------------------------------------------------------------
# 可选模型（gpt-image-2 系列）。不同模型的 quality 档位不同：
#   - gpt-image-2           : auto/low/medium/high
#   - gpt-image-2.5-flare   : 快速·高质量日常生图，档位多出 xhigh/max
#   - gpt-image-2.5-sunburst: 高精度图片编辑，档位多出 xhigh/max
# 模型 ID 与 quality 档位来自 AiHubMix 官方更新日志（2026-09-09）。
DEFAULT_MODEL = "gpt-image-2"
MODEL_SPECS = {
    "gpt-image-2": {
        "label": "GPT Image 2（稳定版）",
        "qualities": ["auto", "low", "medium", "high"],
        "default_quality": "medium",
    },
    "gpt-image-2.5-flare": {
        "label": "GPT Image 2.5 Flare（快速·高质量日常生图）",
        "qualities": ["auto", "low", "medium", "high", "xhigh", "max"],
        "default_quality": "medium",
    },
    "gpt-image-2.5-sunburst": {
        "label": "GPT Image 2.5 Sunburst（高精度图片编辑）",
        "qualities": ["auto", "low", "medium", "high", "xhigh", "max"],
        "default_quality": "medium",
    },
}
MODEL = DEFAULT_MODEL                 # 向后兼容别名：默认模型
ALL_QUALITIES = ["auto", "low", "medium", "high", "xhigh", "max"]  # argparse 并集，按模型再校验
MAX_REF_BYTES = 1_500_000             # 单张参考图上传体积上限（超过后服务端解码会挂死）
REF_MAX_EDGE_NO_LIMIT = True          # 只重编码、绝不改像素
UPLOAD_WINDOW = 180                   # socket 第一段：连接 + 上传请求体的超时（秒）
READ_WINDOW = 240                     # socket 第二段：等服务端出图的读超时（秒）
                                      # 实测：成功都在 20~150s 返回；挂死会一直静默到上限
HARD_CAP = UPLOAD_WINDOW + READ_WINDOW + 30   # 线程级总时长兜底（防线程挂死）

# --- 原生异步协议（/ai/v1）常量（2026-09-22 实测验证） ---
ASYNC_POLL_INTERVAL = 15              # 官方推荐每 15s 轮询一次任务状态
ASYNC_POLL_WINDOW = 30 * 60           # 本地轮询窗口（秒）；耗尽抛 TimeoutError 但任务 ID 保留可恢复
ASYNC_ARTIFACT_TTL = 2 * 3600         # 官方：产物保留 2 小时，过期下载返回 410 artifact_expired
ASYNC_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}

# (aspect_ratio, resolution) -> size 映射（像素，均为 16 的倍数）
# gpt-image-2 系列（含 2.5）共用：官方约束「宽高均被 16 整除、比例 1:3~3:1」对该系列通用
SIZE_MAP = {
    "4K": {
        "1:1": "2880x2880", "16:9": "3840x2160", "9:16": "2160x3840",
        "4:3": "2880x2160", "3:4": "2160x2880", "3:2": "3072x2048",
        "2:3": "2048x3072", "21:9": "3840x1648", "5:4": "3072x2448",
    },
    "2K": {
        "1:1": "2048x2048", "16:9": "2048x1152", "9:16": "1152x2048",
        "4:3": "2048x1536", "3:4": "1536x2048", "3:2": "2048x1360",
        "2:3": "1360x2048", "21:9": "2048x880", "5:4": "2048x1632",
    },
    "1K": {
        "1:1": "1024x1024", "16:9": "1536x1024", "9:16": "1024x1536",
        "4:3": "1536x1152", "3:4": "1152x1536", "3:2": "1536x1024",
        "2:3": "1024x1536", "21:9": "1536x656", "5:4": "1280x1024",
    },
}

# 官方 size 约束（gpt-image-2）：支持任意 WIDTHxHEIGHT，但宽高均须被 16 整除、
# 宽高比须在 1:3 ~ 3:1 之间；超过 2560x1440 的分辨率官方标记为实验性（可能失败）。


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def validate_size(size: str) -> str:
    """按官方约束校验 size：宽高均被 16 整除、宽高比 1:3~3:1；>2560x1440 仅警告。"""
    try:
        w, h = (int(v) for v in size.lower().split("x"))
        assert w > 0 and h > 0
    except Exception:
        raise ValueError(f"size 格式错误：{size}（应为 WIDTHxHEIGHT，如 1536x864）")
    if w % 16 or h % 16:
        raise ValueError(f"size {size} 违反官方约束：宽高都必须能被 16 整除")
    if not (1 / 3 <= w / h <= 3):
        raise ValueError(f"size {size} 宽高比超出官方允许范围 1:3 ~ 3:1")
    if w * h > 2560 * 1440:
        log(f"警告：{size} 超过 2560x1440，官方标记为实验性分辨率，可能被拒或不稳定")
    return size


def resolve_size(pixel_size: str, ratio: str, resolution: str) -> str:
    if pixel_size:
        return validate_size(pixel_size)
    res = (resolution or "2K").upper()
    table = SIZE_MAP.get(res, SIZE_MAP["2K"])
    return validate_size(table.get(ratio, table.get("16:9")))


def get_model_spec(model: str) -> dict:
    """取模型规格；未知模型直接报错，避免把错误 model ID 发给上游白白计费。"""
    spec = MODEL_SPECS.get(model)
    if not spec:
        raise ValueError(f"未知模型：{model}（可选：{', '.join(MODEL_SPECS)}）")
    return spec


def validate_quality(model: str, quality: str) -> str:
    """校验 quality 是否被该模型支持（2.5 系列多出 xhigh/max，gpt-image-2 不支持）。"""
    spec = get_model_spec(model)
    if quality not in spec["qualities"]:
        raise ValueError(
            f"模型 {model} 不支持 quality={quality}"
            f"（可选：{', '.join(spec['qualities'])}）"
        )
    return quality


def load_config(args):
    """按优先级组装 api_key / api_base(/v1) / proxy。"""
    # --- api_key & api_base ---
    api_key = args.api_key or os.getenv("AIHUBMIX_API_KEY")
    api_base = args.api_base or os.getenv("AIHUBMIX_API_BASE")

    if not api_key or not api_base:
        db = Path(__file__).resolve().parent.parent / "backend" / "instance" / "database.db"
        if db.exists():
            try:
                conn = sqlite3.connect(str(db))
                cur = conn.cursor()
                cur.execute("PRAGMA table_info(settings)")
                cols = [r[1] for r in cur.fetchall()]
                cur.execute("SELECT * FROM settings LIMIT 1")
                row = cur.fetchone()
                conn.close()
                if row:
                    d = dict(zip(cols, row))
                    api_key = api_key or d.get("api_key")
                    api_base = api_base or d.get("api_base_url")
            except Exception as e:
                log(f"读取 settings 数据库失败（忽略）：{e}")

    # 归一化到 /v1（gpt-image-2 必须走 /v1 路由，不能走 /gemini）
    base = (api_base or "https://aihubmix.com/v1").rstrip("/")
    if base.endswith("/gemini"):
        base = base[: -len("/gemini")] + "/v1"
    if not base.endswith("/v1"):
        base = base + "/v1"

    # --- proxy ---
    proxy = args.proxy or os.getenv("AIHUBMIX_PROXY")
    if proxy is None:
        proxy = _detect_system_proxy()

    return api_key, base, proxy


def _detect_system_proxy():
    """探测系统代理：先试直连 aihubmix，通则不用代理，否则回退系统代理。"""
    import urllib3
    urllib3.disable_warnings()
    try:
        import urllib.request
        probe = "https://aihubmix.com"
        # 1) 直连（忽略环境变量与证书）
        try:
            s = requests.Session()
            s.trust_env = False
            s.get(probe, timeout=(5, 8), verify=False,
                  proxies={"http": None, "https": None})
            log("[net] 直连 aihubmix 可用，不使用代理")
            return None
        except Exception:
            pass
        # 2) 系统代理
        sys_proxies = urllib.request.getproxies()
        p = sys_proxies.get("https") or sys_proxies.get("http")
        if p and not p.lower().startswith("http"):
            p = "http://" + p
        if p:
            log(f"[net] 使用系统代理：{p}")
            return p
    except Exception as e:
        log(f"[net] 代理探测异常（忽略）：{e}")
    log("[net] 未探测到可用代理，直连")
    return None


def encode_ref_for_upload(path: str, max_edge: int = None):
    """把参考图压到 MAX_REF_BYTES 以内，默认只换编码（PNG->WebP），像素尺寸绝不改变。

    max_edge: 可选。传入时先把最长边缩到该值以内（会改输入像素，仅当用户显式
              开启「压缩参考图像素」时使用，用于提高上游多图请求成功率）。
    Returns: (bytes, mime, filename, note)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"参考图不存在：{path}")
    orig = p.stat().st_size
    suffix = p.suffix.lower().lstrip(".") or "png"

    img = Image.open(p)
    w, h = img.size
    shrunk = ""
    if max_edge and max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        nw, nh = max(16, int(w * scale) // 16 * 16), max(16, int(h * scale) // 16 * 16)
        img = img.resize((nw, nh), Image.LANCZOS)
        shrunk = f"，像素 {w}x{h} -> {nw}x{nh}（用户开启压缩）"
        w, h = nw, nh

    if orig <= MAX_REF_BYTES and not shrunk:
        mime = {"png": "image/png", "webp": "image/webp",
                "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix, "image/png")
        return p.read_bytes(), mime, f"ref.{suffix}", f"{orig/1048576:.2f}MB 原样上传"

    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    payload, used_q = None, None
    for q in (92, 88, 84, 80, 75, 70, 65, 60, 55, 50):
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=q, method=4)
        payload = buf.getvalue()
        used_q = q
        if len(payload) <= MAX_REF_BYTES:
            break

    note = (f"{orig/1048576:.2f}MB -> {len(payload)/1048576:.2f}MB (WebP q{used_q})"
            f"{shrunk or f'，像素 {w}x{h} 未改变'}")
    if len(payload) > MAX_REF_BYTES:
        note += "，警告：最低画质仍超 1.5MB"
    return payload, "image/webp", "ref.webp", note


def do_request(base, api_key, prompt, size, ref_paths, quality, proxy, ref_max_edge=None,
               background=None, output_format=None, model=DEFAULT_MODEL):
    """发一次请求。无参考图走 /images/generations，有参考图走 /images/edits。

    background/output_format 仅在显式指定时发送：透明背景为官方预览功能，
    需 background=transparent + output_format=png/webp，返回带真实 alpha 的图。
    """
    session = requests.Session()
    session.trust_env = False
    proxies = {"http": proxy, "https": proxy} if proxy else {"http": None, "https": None}
    headers = {"Authorization": f"Bearer {api_key}"}
    t0 = time.time()

    if ref_paths:
        url = f"{base}/images/edits"
        files, total = [], 0
        for i, rp in enumerate(ref_paths):
            data_bytes, mime, fname, note = encode_ref_for_upload(rp, ref_max_edge)
            total += len(data_bytes)
            log(f"  参考图{i+1}: {note}")
            files.append(("image[]", (f"ref_{i}_{fname}", io.BytesIO(data_bytes), mime)))
        log(f"POST {url}  参考图 {len(files)} 张 / 实际上传 {total/1048576:.2f} MB  "
            f"size={size} background={background or '-'} format={output_format or '-'}  "
            f"上传窗口={UPLOAD_WINDOW}s 读窗口={READ_WINDOW}s")
        data = {"model": model, "prompt": prompt, "n": "1", "size": size}
        if background:
            data["background"] = background
        if output_format:
            data["output_format"] = output_format
        try:
            r = session.post(url, headers=headers, data=data, files=files,
                             timeout=(UPLOAD_WINDOW, READ_WINDOW), proxies=proxies, verify=False)
            log(f"/images/edits 收到响应，耗时 {time.time()-t0:.1f}s  status={r.status_code}")
            return r
        except Exception as e:
            log(f"/images/edits 失败于 {time.time()-t0:.1f}s: {type(e).__name__}")
            raise
        finally:
            for _, fo in files:
                try:
                    fo[1].close()
                except Exception:
                    pass
    else:
        url = f"{base}/images/generations"
        payload = {"model": model, "prompt": prompt, "n": 1, "size": size, "quality": quality}
        if background:
            payload["background"] = background
        if output_format:
            payload["output_format"] = output_format
        hdrs = dict(headers)
        hdrs["Content-Type"] = "application/json"
        if quality in ("high", "xhigh", "max"):
            log(f"警告：quality={quality} 出图慢（实测 high 常 >60s，xhigh/max 更甚），"
                "而代理/网关链路约 60s 空闲即掐断，本次请求可能超时失败"
                "（gpt-image-2 medium 出 4K 约 54s，在窗口内）")
        log(f"POST {url}  无参考图  size={size} quality={quality} "
            f"background={background or '-'} format={output_format or '-'}  读窗口={READ_WINDOW}s")
        try:
            r = session.post(url, headers=hdrs, json=payload,
                             timeout=(UPLOAD_WINDOW, READ_WINDOW), proxies=proxies, verify=False)
            log(f"/images/generations 收到响应，耗时 {time.time()-t0:.1f}s  status={r.status_code}")
            return r
        except Exception as e:
            log(f"/images/generations 失败于 {time.time()-t0:.1f}s: {type(e).__name__}")
            raise


def call_with_retry(base, api_key, prompt, size, ref_paths, quality, proxy, max_retries,
                    ref_max_edge=None, retry_on_read_timeout=False,
                    background=None, output_format=None, model=DEFAULT_MODEL):
    """线程池包裹 + 分层重试。

    - ReadTimeout：默认绝不重试（请求体已送达、服务端可能已计费；且挂死重试也是白花钱）。
      仅当用户显式开启 retry_on_read_timeout 时，挂死才允许重试 1 次（接受可能重复计费）。
    - ProxyError / ConnectionError（含 write timed out，请求体不完整会被丢弃）：可安全重试
    """
    attempt = 0
    hang_retried = False
    while True:
        attempt += 1
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(do_request, base, api_key, prompt, size,
                                     ref_paths, quality, proxy, ref_max_edge,
                                     background, output_format, model)
            try:
                return future.result(timeout=HARD_CAP)
            except FutureTimeout:
                future.cancel()
                raise TimeoutError(f"总时长超过 {HARD_CAP}s，判定为服务端挂死")
        except requests.exceptions.ReadTimeout:
            if retry_on_read_timeout and not hang_retried:
                hang_retried = True
                log("读超时（上游挂死）。用户已开启「挂死自动重试」，重试 1 次（可能重复计费）...")
                time.sleep(3)
                continue
            log("读超时：服务端在读窗口内未返回（上游 gpt-image 挂死）。不重试以避免重复计费。")
            raise
        except (requests.exceptions.ProxyError, requests.exceptions.ConnectionError) as e:
            if _disconnected_while_waiting(e):
                # 2026-09-21 实测：同步 ~60s 断连的异常链终点是 RemoteDisconnected，
                # 位于 conn.getresponse() —— 请求体已送达上游、可能已受理计费，
                # 绝不能当「未送达」自动重试（会重复花钱）。
                log("连接在等待响应阶段被对端关闭（请求已送达上游，可能已计费）：不自动重试。")
                raise
            if attempt <= max_retries:
                # 连接类失败可能是当前链路（代理/直连）临时坏了：重新探测，变了就换路
                alt = _detect_system_proxy()
                if alt != proxy:
                    log(f"换链路重试：{proxy or '直连'} -> {alt or '直连'}")
                    proxy = alt
                log(f"请求未送达（{type(e).__name__}），{attempt}/{max_retries} 次重试...")
                time.sleep(min(3 * attempt, 6))
                continue
            raise
        finally:
            executor.shutdown(wait=False)


def _disconnected_while_waiting(error: BaseException) -> bool:
    """判断连接类异常是否发生在「请求已发出、等待响应」阶段（而非建连阶段）。

    沿 __cause__/__context__ 异常链找 RemoteDisconnected / BadStatusLine /
    ConnectionResetError：命中即视为上游可能已受理，不安全，禁止自动重试。
    """
    seen, cur = set(), error
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if type(cur).__name__ in ("RemoteDisconnected", "BadStatusLine", "ConnectionResetError"):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


# ----------------------------------------------------------------------------
# AiHubMix 原生异步协议（/ai/v1/images/generations + /ai/v1/tasks/*）
# 实测（2026-09-22）：创建约 2s 返回任务 ID；max 画质 2160x3840 约 76s completed；
# 产物为二进制直接下载（跟随重定向），保留 2 小时。所有函数绝不自动重试创建。
# ----------------------------------------------------------------------------
def native_origin(base: str) -> str:
    """从 OpenAI 兼容 base（.../v1）推出原生协议 origin（https://aihubmix.com）。"""
    return base[: -len("/v1")] if base.endswith("/v1") else base


def async_session(api_key: str, proxy: str = None) -> requests.Session:
    """异步协议专用 Session：显式代理、max_retries=0（任何一层都不许悄悄重发）。"""
    s = requests.Session()
    s.trust_env = False
    s.headers["Authorization"] = f"Bearer {api_key}"
    s.proxies = {"http": proxy, "https": proxy} if proxy else {"http": None, "https": None}
    s.verify = False
    adapter = requests.adapters.HTTPAdapter(max_retries=0, pool_maxsize=4)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def build_async_payload(model, prompt, size, quality, ref_paths=None, ref_max_edge=None,
                        background=None, output_format=None) -> dict:
    """组装原生异步创建请求体。

    - quality/background 按官方约定放 extra（2026-09-22 实测通过）。
    - 参考图以 data URI 放 images[]（官方 schema：image/images 接受 URL / data URI /
      raw base64，三模型 gpt-image-2 / 2.5-flare / 2.5-sunburst 均已确认支持）。
      注意 base64 会使体积膨胀 4/3，整请求上限 32 MiB。
    """
    payload = {"model": model, "prompt": prompt, "n": 1, "size": size, "async": True}
    extra = {}
    if quality:
        extra["quality"] = quality
    if background:
        extra["background"] = background
    if extra:
        payload["extra"] = extra
    if output_format:
        payload["output_format"] = output_format
    if ref_paths:
        images = []
        for i, rp in enumerate(ref_paths):
            data_bytes, mime, _fname, note = encode_ref_for_upload(rp, ref_max_edge)
            log(f"  参考图{i+1}: {note}")
            b64 = base64.b64encode(data_bytes).decode("ascii")
            images.append(f"data:{mime};base64,{b64}")
        payload["images"] = images
    return payload


def async_create(session, base, payload) -> dict:
    """创建异步任务，返回任务 JSON（含 id/status）。

    网络异常时直接抛出、绝不重试：创建请求可能已被上游受理并计费，
    调用方必须记录「状态未知」，用同一把 Key 查 GET /ai/v1/tasks 列表找回，
    确认未受理后才允许人工重新提交。
    """
    url = native_origin(base) + "/ai/v1/images/generations"
    log(f"POST {url}  异步创建  model={payload.get('model')} size={payload.get('size')} "
        f"quality={(payload.get('extra') or {}).get('quality', '-')} "
        f"images={len(payload.get('images') or [])} 张")
    r = session.post(url, json=payload, timeout=(60, 120), allow_redirects=False)
    if r.status_code >= 400 or not r.content:
        raise RuntimeError(f"异步创建失败 HTTP {r.status_code}: {r.text[:600]}")
    task = r.json()
    if not task.get("id"):
        raise RuntimeError(f"异步创建响应缺少任务 ID：{str(task)[:400]}")
    log(f"异步任务已创建 id={task['id']} status={task.get('status')}")
    return task


def async_status(session, base, task_id) -> dict:
    """查询任务状态 GET /ai/v1/tasks/{id}（幂等 GET，可安全重复）。"""
    url = native_origin(base) + f"/ai/v1/tasks/{task_id}"
    r = session.get(url, timeout=(15, 60))
    if r.status_code >= 400:
        raise RuntimeError(f"查询任务失败 HTTP {r.status_code}: {r.text[:400]}")
    return r.json()


def async_extract(session, base, task_id, result_id=None) -> bytes:
    """下载产物 GET /ai/v1/tasks/{id}/content[/{result_id}]（跟随重定向，幂等）。

    多产物缺 result_id 时上游返回 400 result_id_required，此时从任务详情取
    output[0] 的 result_id 再试；产物过期（2 小时）返回 410。
    """
    url = native_origin(base) + f"/ai/v1/tasks/{task_id}/content"
    if result_id:
        url += f"/{result_id}"
    r = session.get(url, timeout=(30, 300), allow_redirects=True)
    if r.status_code == 400 and b"result_id_required" in (r.content or b""):
        task = async_status(session, base, task_id)
        outputs = task.get("output") or []
        if not outputs:
            raise RuntimeError(f"result_id_required 但任务无 output：{str(task)[:300]}")
        rid = outputs[0].get("id") or outputs[0].get("result_id")
        if not rid:
            raise RuntimeError(f"无法从 output 取得 result_id：{str(outputs[0])[:300]}")
        log(f"单结果约定不适用，改用 result_id={rid} 下载")
        return async_extract(session, base, task_id, rid)
    if r.status_code == 410:
        raise RuntimeError(f"产物已过期（410）：任务 {task_id} 的产物仅保留 2 小时，无法再下载")
    if r.status_code >= 400:
        raise RuntimeError(f"下载产物失败 HTTP {r.status_code}: {r.text[:400]}")
    return r.content


def async_wait_and_extract(session, base, task_id, poll_window=ASYNC_POLL_WINDOW,
                           poll_interval=ASYNC_POLL_INTERVAL, on_status=None):
    """轮询到终态并下载产物，返回 (image_bytes, task_detail)。

    - on_status(status, task)：每轮回调（调用方据此持久化进度）。
    - 查询是幂等 GET：单次网络失败不放弃，等下一轮。
    - 轮询窗口耗尽抛 TimeoutError（带任务 ID）：任务仍在上游运行，
      之后可凭 ID 继续查询/下载（产物保留 2 小时），绝不重新创建。
    """
    deadline = time.time() + poll_window
    task, status = {}, "unknown"
    while True:
        try:
            task = async_status(session, base, task_id)
            status = task.get("status") or "unknown"
        except Exception as e:
            status = "poll_error"
            log(f"查询任务临时失败（下一轮重试）：{type(e).__name__}: {str(e)[:200]}")
        if on_status:
            try:
                on_status(status, task)
            except Exception:
                pass
        if status in ASYNC_TERMINAL_STATUSES:
            break
        if time.time() >= deadline:
            raise TimeoutError(
                f"任务 {task_id} 在 {poll_window}s 内未到终态（最后状态 {status}）；"
                f"任务仍在运行，之后可凭任务 ID 查询/下载（产物保留 2 小时）")
        time.sleep(poll_interval)

    if status == "failed":
        err = task.get("error") or {}
        raise RuntimeError(f"任务失败 code={err.get('code')}: "
                           f"{err.get('message') or str(task)[:300]}")
    if status == "cancelled":
        raise RuntimeError(f"任务被取消：{task_id}")
    log(f"任务 {task_id} completed，下载产物...")
    data = async_extract(session, base, task_id)
    return data, task


def parse_response_to_image(resp) -> Image.Image:
    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text[:800]}")
    j = resp.json()
    items = j.get("data") or []
    if not items:
        raise RuntimeError(f"响应中无图片数据：{str(j)[:500]}")
    item = items[0]
    if item.get("b64_json"):
        raw = base64.b64decode(item["b64_json"])
        return Image.open(io.BytesIO(raw))
    if item.get("url"):
        log(f"响应返回 URL，下载中：{item['url'][:80]}...")
        r = requests.get(item["url"], timeout=(15, 120))
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content))
    raise RuntimeError(f"响应条目既无 b64_json 也无 url：{str(item)[:300]}")


def main():
    ap = argparse.ArgumentParser(description="独立 GPT Image 出图程序（gpt-image-2 / 2.5）")
    ap.add_argument("--prompt", required=True, help="生成提示词")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODEL_SPECS),
                    help="出图模型；默认 gpt-image-2，另支持 gpt-image-2.5-flare / gpt-image-2.5-sunburst")
    ap.add_argument("--ref", nargs="*", default=[], help="参考图路径（0~多张），带参考图走图生图")
    ap.add_argument("--size", default="2K", choices=["1K", "2K", "4K"], help="分辨率档位")
    ap.add_argument("--ratio", default="16:9", help="宽高比，如 16:9 / 9:16 / 1:1")
    ap.add_argument("--pixel-size", default="", help="直接指定像素 size，如 2160x3840（覆盖 --size/--ratio）")
    ap.add_argument("--background", default="", choices=["", "auto", "transparent", "opaque"],
                    help="背景透明度；transparent 为官方预览功能（不额外收费），返回带真实 alpha 的图")
    ap.add_argument("--output-format", default="", choices=["", "png", "webp", "jpeg"],
                    help="输出格式；--background transparent 时必须为 png 或 webp")
    ap.add_argument("--quality", default=None, choices=ALL_QUALITIES,
                    help="画质档位；不同模型支持范围不同（2.5 系列多出 xhigh/max）。"
                         "留空则取该模型推荐值 medium；high/xhigh/max 更慢更易超时")
    ap.add_argument("--out", default="", help="输出文件路径，默认 image2/out_<时间戳>.png")
    ap.add_argument("--retries", type=int, default=1, help="安全错误（连接类）最大重试次数（仅 --sync 路径）")
    ap.add_argument("--retry-on-hang", action="store_true",
                    help="（仅 --sync）读超时（上游挂死）时自动重试 1 次；注意：挂死的请求可能已计费，可能重复花钱")
    ap.add_argument("--sync", action="store_true",
                    help="回退旧同步 /v1/images/* 路径；链路空闲 >60s 即断，长生成大概率失败，仅作备用")
    ap.add_argument("--poll-window", type=int, default=ASYNC_POLL_WINDOW,
                    help="异步轮询窗口（秒），默认 1800；耗尽后任务 ID 仍可用于恢复")
    ap.add_argument("--ref-max-edge", type=int, default=0,
                    help="把参考图最长边缩到该值以内（如 2048），会改输入像素，仅建议多图失败时启用")
    ap.add_argument("--api-key", default="", help="覆盖 API key")
    ap.add_argument("--api-base", default="", help="覆盖 API base")
    ap.add_argument("--proxy", default=None, help="覆盖代理，如 http://127.0.0.1:10809；传空串表示强制直连")
    args = ap.parse_args()

    api_key, base, proxy = load_config(args)
    if not api_key:
        log("错误：未找到 API key（--api-key / AIHUBMIX_API_KEY / settings 数据库均为空）")
        sys.exit(2)
    if args.proxy == "":
        proxy = None  # 显式传 --proxy "" 表示强制直连

    # 校验模型与画质档位（quality 留空则取该模型默认值）
    try:
        get_model_spec(args.model)
        quality = args.quality or MODEL_SPECS[args.model]["default_quality"]
        validate_quality(args.model, quality)
        size = resolve_size(args.pixel_size, args.ratio, args.size)
    except ValueError as e:
        log(f"错误：{e}")
        sys.exit(2)

    background = args.background or None
    output_format = args.output_format or None
    if background == "transparent":
        if output_format == "jpeg":
            log("错误：background=transparent 时官方要求输出格式为 png 或 webp（jpeg 无 alpha 通道）")
            sys.exit(2)
        if not output_format:
            output_format = "png"
            log("透明背景需带 alpha 的格式承载，output_format 自动取 png")

    masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    log(f"model={args.model}  base={base}  key={masked}  proxy={proxy}")
    log(f"size={size}  参考图={len(args.ref)} 张  quality={quality}  "
        f"background={background or '-'}  output_format={output_format or '-'}")

    t0 = time.time()
    try:
        if args.sync:
            log("已指定 --sync：走旧同步 /v1/images/* 路径（>60s 的生成大概率被链路掐断）")
            resp = call_with_retry(base, api_key, args.prompt, size,
                                   args.ref, quality, proxy, args.retries,
                                   ref_max_edge=args.ref_max_edge or None,
                                   retry_on_read_timeout=args.retry_on_hang,
                                   background=background, output_format=output_format,
                                   model=args.model)
            img = parse_response_to_image(resp)
        else:
            session = async_session(api_key, proxy)
            payload = build_async_payload(args.model, args.prompt, size, quality,
                                          args.ref, args.ref_max_edge or None,
                                          background, output_format)
            try:
                task = async_create(session, base, payload)
            except requests.exceptions.RequestException as e:
                log(f"创建阶段网络异常：{type(e).__name__}: {str(e)[:300]}")
                log("任务状态未知：不自动重发（可能重复计费）。"
                    "请先用同一把 Key 查 GET /ai/v1/tasks 列表确认是否已受理。")
                sys.exit(1)
            task_id = task["id"]
            log(f"任务 ID：{task_id}（产物保留 2 小时，恢复下载："
                f"GET {native_origin(base)}/ai/v1/tasks/{task_id}/content）")
            try:
                data, task = async_wait_and_extract(session, base, task_id,
                                                    poll_window=args.poll_window)
            except requests.exceptions.RequestException as e:
                log(f"轮询/下载阶段网络异常：{type(e).__name__}: {str(e)[:200]}")
                log(f"任务 {task_id} 可恢复：稍后 GET /ai/v1/tasks/{task_id}/content 下载（2 小时内）")
                sys.exit(1)
            img = Image.open(io.BytesIO(data))
            img.load()
            usage = (task or {}).get("usage") or {}
            if usage.get("cost") is not None:
                log(f"usage: {usage}")
    except Exception as e:
        log(f"失败：{type(e).__name__}: {str(e)[:400]}")
        sys.exit(1)

    ext = {"png": ".png", "webp": ".webp", "jpeg": ".jpg"}.get(output_format or "png", ".png")
    out = args.out or str(Path(__file__).resolve().parent / f"out_{int(time.time())}{ext}")
    if output_format == "jpeg" and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")  # jpeg 无 alpha 通道，先转 RGB 再存
    img.save(out)
    alpha = "带 alpha" if img.mode in ("RGBA", "LA", "PA") else "无 alpha"
    log(f"成功！总耗时 {time.time()-t0:.1f}s  输出尺寸 {img.size} ({alpha})  已保存：{out}")
    if background == "transparent" and img.mode not in ("RGBA", "LA", "PA"):
        log("警告：要求了透明背景，但返回图无 alpha 通道（网关可能未透传 background 参数）")


if __name__ == "__main__":
    main()
