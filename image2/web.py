"""
image2 的本地 Web 界面 —— 浏览器里直接用 AiHubMix GPT Image 2 / 2.5 出图。

启动：
    uv run python image2/web.py
    （默认 http://127.0.0.1:5001 ，可用 --port 改端口）

本文件只是 gen.py 的一层薄封装：
    - 生成默认走 AiHubMix 原生异步协议（创建任务 → 轮询 → 下载产物），
      彻底绕开同步链路「约 60s 空闲即断」的问题（2026-09-22 实测 max 4K 约 76s 出图）
    - 任务记录持久化到 web_outputs/tasks/*.json：服务重启后，未完成任务凭
      供应商任务 ID 自动恢复轮询/下载（产物保留 2 小时）；浏览器刷新也能续查
    - 创建阶段网络异常 = 状态未知：绝不自动重发（可能重复计费），
      提示先用同一把 Key 查 GET /ai/v1/tasks 列表确认是否已受理
"""
import argparse
import io
import json
import os
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import requests
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

import gen  # 复用已跑通的出图逻辑（同步 + 异步协议原语都在 gen.py）

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "web_outputs"
OUT_DIR.mkdir(exist_ok=True)
TASKS_DIR = OUT_DIR / "tasks"          # 任务记录持久化目录（重启恢复的依据）
TASKS_DIR.mkdir(exist_ok=True)

app = Flask(__name__)

# 任务表：task_id -> {status, vendor_task_id, vendor_status, error, filename,
#                     img_size, elapsed, created, updated, params}
# 内存只是镜像，磁盘 web_outputs/tasks/{task_id}.json 才是权威。
TASKS = {}
TASKS_LOCK = threading.Lock()

# 非终态（重启时需要恢复跟踪的状态）
RESUMABLE_STATUSES = {"pending", "creating", "submitted", "running"}

RATIOS = list(gen.SIZE_MAP["2K"].keys())


# ----------------------------------------------------------------------------
# 任务记录持久化
# ----------------------------------------------------------------------------
def _task_path(task_id: str) -> Path:
    return TASKS_DIR / f"{task_id}.json"


def _save_task(rec: dict):
    """原子落盘任务记录（tmp + replace，避免半截 JSON）。调用方需持锁或保证单线程写。"""
    p = _task_path(rec["task_id"])
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(p))


def _update_task(task_id: str, **fields):
    """更新内存 + 磁盘任务记录。持久化失败只记日志，不打断生成流程。"""
    with TASKS_LOCK:
        rec = TASKS.get(task_id)
        if rec is None:
            return
        rec.update(fields)
        rec["updated"] = time.time()
        try:
            _save_task(rec)
        except Exception as e:
            gen.log(f"[web:{task_id[:8]}] 任务记录持久化失败（不影响生成）：{e}")


# ----------------------------------------------------------------------------
# 生成线程（异步协议）
# ----------------------------------------------------------------------------
def _save_result_image(task_id: str, data: bytes, task: dict, params: dict, t0: float):
    """把下载到的产物字节存盘并标记 done。

    额外检测产物是否真带透明通道，用于在 web 端把「透明背景是否生效」直接显示出来
    （替代单独跑 CLI 验证、且不产生任何额外 API 费用）：
      - has_alpha    ：图像模式含 alpha 通道（RGBA/LA/PA）
      - transparent_ok：确有透明像素（alpha 通道最小值 < 255），即真正抠出了背景
    """
    output_format = params.get("output_format") or "png"
    ext = {"png": "png", "webp": "webp", "jpeg": "jpg"}.get(output_format, "png")
    fname = f"{task_id}.{ext}"
    img = Image.open(io.BytesIO(data))
    img.load()  # 强制解码校验，坏图在这里报错而不是存下半截文件

    # 透明通道检测：必须在 jpeg 转换（会丢弃 alpha）之前做
    has_alpha = img.mode in ("RGBA", "LA", "PA")
    alpha_min = None
    if has_alpha:
        try:
            alpha_min = img.getchannel("A").getextrema()[0]
        except Exception:
            alpha_min = None
    transparent_ok = bool(has_alpha and alpha_min is not None and alpha_min < 255)

    if output_format == "jpeg" and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")  # jpeg 无 alpha 通道，先转 RGB 再存
    img.save(str(OUT_DIR / fname))
    usage = (task or {}).get("usage") or {}
    _update_task(task_id, status="done", filename=fname, error="",
                 img_size=f"{img.size[0]}x{img.size[1]}",
                 elapsed=round(time.time() - t0, 1),
                 vendor_status="completed",
                 usage_cost=usage.get("cost"),
                 has_alpha=has_alpha, transparent_ok=transparent_ok)
    if has_alpha:
        alpha_note = f"带alpha(min={alpha_min},{'有透明像素' if transparent_ok else '全不透明'})"
    else:
        alpha_note = "无alpha"
    gen.log(f"[web:{task_id[:8]}] 成功 {img.size} {alpha_note} 本轮耗时 {time.time()-t0:.1f}s")


def _poll_and_finish(task_id: str, session, base: str, vendor_id: str, params: dict):
    """轮询供应商任务到终态并下载产物。查询/下载均为幂等 GET，可安全反复执行。"""
    t0 = time.time()

    def on_status(status, _task):
        if status in ("pending", "in_progress"):
            _update_task(task_id, status="running", vendor_status=status)
        # poll_error：gen.async_wait_and_extract 内部已容错（等下一轮），这里不动状态

    data, task = gen.async_wait_and_extract(session, base, vendor_id, on_status=on_status)
    _save_result_image(task_id, data, task, params, t0)


def _run_task(task_id: str, params: dict):
    """完整生成流程：解析配置 → 组装 payload → 异步创建 → 轮询 → 下载存盘。

    params: prompt/ratio/size_tier/quality/pixel_size/background/output_format/
            model/ref_max_edge/ref_saved(临时参考图路径列表，用完即删)
    """
    ref_saved = params.get("ref_saved") or []
    try:
        ns = SimpleNamespace(api_key="", api_base="", proxy=None)
        api_key, base, proxy = gen.load_config(ns)
        if not api_key:
            raise RuntimeError("未找到 API key（settings 数据库 / 环境变量均为空）")

        size = gen.resolve_size(params.get("pixel_size") or "",
                                params.get("ratio") or "16:9",
                                params.get("size_tier") or "2K")
        session = gen.async_session(api_key, proxy)
        payload = gen.build_async_payload(
            params["model"], params["prompt"], size, params.get("quality"),
            ref_saved, params.get("ref_max_edge"),
            params.get("background") or None, params.get("output_format") or None)
        gen.log(f"[web:{task_id[:8]}] model={params['model']} base={base} size={size} "
                f"ref={len(ref_saved)} quality={params.get('quality')} "
                f"background={params.get('background') or '-'} "
                f"format={params.get('output_format') or '-'}")

        _update_task(task_id, status="creating")
        try:
            task = gen.async_create(session, base, payload)
        except requests.exceptions.RequestException as e:
            # 创建阶段断网 = 上游可能已受理并计费：绝不自动重发（money 规则）
            raise RuntimeError(
                f"创建阶段网络异常（{type(e).__name__}），任务状态未知：不会自动重发。"
                f"请先用同一把 Key 查 GET {gen.native_origin(base)}/ai/v1/tasks 列表"
                f"确认上游是否已受理，再决定是否重新提交") from e
        vendor_id = task["id"]
        _update_task(task_id, status="submitted", vendor_task_id=vendor_id,
                     vendor_status=task.get("status") or "")
        gen.log(f"[web:{task_id[:8]}] 供应商任务 ID：{vendor_id}"
                f"（产物保留 2 小时，重启后可自动恢复）")

        _poll_and_finish(task_id, session, base, vendor_id, params)
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:600]}"
        gen.log(f"[web:{task_id[:8]}] 失败 {err}")
        _update_task(task_id, status="error", error=err)
    finally:
        # 清理上传的临时参考图
        for p in ref_saved:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass


def _resume_worker(task_id: str, vendor_id: str, params: dict):
    """重启恢复线程：凭供应商任务 ID 继续轮询/下载（不重新创建，不会重复计费）。"""
    try:
        ns = SimpleNamespace(api_key="", api_base="", proxy=None)
        api_key, base, proxy = gen.load_config(ns)
        if not api_key:
            raise RuntimeError("未找到 API key，无法恢复任务")
        session = gen.async_session(api_key, proxy)
        gen.log(f"[web:{task_id[:8]}] 恢复跟踪供应商任务 {vendor_id}")
        _poll_and_finish(task_id, session, base, vendor_id, params)
    except Exception as e:
        err = f"恢复失败 {type(e).__name__}: {str(e)[:400]}"
        gen.log(f"[web:{task_id[:8]}] {err}")
        _update_task(task_id, status="error", error=err)


def _resume_tasks():
    """启动时扫描持久化任务：非终态且未超产物保留期（2h）的任务自动恢复轮询。

    - 无供应商任务 ID（创建阶段崩溃）：标记错误，提示查 /ai/v1/tasks 列表，绝不重发
    - 超过 2 小时：产物已过期（410），标记错误
    """
    resumed = 0
    for p in sorted(TASKS_DIR.glob("*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        tid = rec.get("task_id") or p.stem
        rec["task_id"] = tid
        with TASKS_LOCK:
            TASKS[tid] = rec
        if rec.get("status") not in RESUMABLE_STATUSES:
            continue
        age = time.time() - (rec.get("created") or 0)
        vendor_id = rec.get("vendor_task_id") or ""
        if not vendor_id:
            _update_task(tid, status="error",
                         error="服务在创建阶段重启，上游是否已受理未知：不会自动重发；"
                               "请用同一把 Key 查 GET /ai/v1/tasks 列表确认后再决定是否重新提交")
        elif age > gen.ASYNC_ARTIFACT_TTL:
            _update_task(tid, status="error",
                         error=f"任务 {vendor_id} 已超过产物保留期（2 小时），产物过期无法恢复；请重新提交")
        else:
            resumed += 1
            threading.Thread(target=_resume_worker,
                             args=(tid, vendor_id, rec.get("params") or {}),
                             daemon=True).start()
    if resumed:
        gen.log(f"[web] 已恢复 {resumed} 个未完成的异步任务")


# ----------------------------------------------------------------------------
# 路由
# ----------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(HERE, "web_index.html")


@app.route("/api/options")
def options():
    models = [{"id": m, "label": spec["label"], "qualities": spec["qualities"],
               "default_quality": spec["default_quality"]}
              for m, spec in gen.MODEL_SPECS.items()]
    return jsonify({"models": models, "ratios": RATIOS, "sizes": ["1K", "2K", "4K"],
                    "backgrounds": ["auto", "transparent", "opaque"],
                    "formats": ["png", "webp", "jpeg"]})


@app.route("/api/generate", methods=["POST"])
def generate():
    prompt = (request.form.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt 不能为空"}), 400
    model = (request.form.get("model") or "").strip() or gen.DEFAULT_MODEL
    try:
        gen.get_model_spec(model)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    ratio = request.form.get("ratio", "16:9")
    size_tier = request.form.get("size", "2K")
    quality = (request.form.get("quality") or "").strip() or gen.MODEL_SPECS[model]["default_quality"]
    try:
        gen.validate_quality(model, quality)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    pixel_size = (request.form.get("pixel_size") or "").strip()
    background = (request.form.get("background") or "").strip() or None
    output_format = (request.form.get("output_format") or "").strip() or None
    if background == "transparent":
        if output_format == "jpeg":
            return jsonify({"error": "透明背景必须使用 png 或 webp 输出（jpeg 无 alpha 通道）"}), 400
        output_format = output_format or "png"
    # 参考图压缩开关（涉及输入像素，默认关闭，绝不默认开启）
    ref_max_edge = 2048 if request.form.get("shrink_refs") == "1" else None

    # 保存上传的参考图到临时位置
    ref_saved = []
    tmp_dir = OUT_DIR / "_tmp"
    tmp_dir.mkdir(exist_ok=True)
    for i, f in enumerate(request.files.getlist("ref_images")):
        if f and f.filename:
            ext = Path(f.filename).suffix.lower() or ".png"
            dest = tmp_dir / f"{uuid.uuid4().hex}_{i}{ext}"
            f.save(str(dest))
            ref_saved.append(str(dest))

    task_id = uuid.uuid4().hex
    now = time.time()
    # 持久化完整 prompt 与全部设置（供「历史记录」一键回填复用）；绝不存参考图 base64
    params_persist = {"prompt": prompt, "model": model, "ratio": ratio,
                      "size_tier": size_tier, "quality": quality, "pixel_size": pixel_size,
                      "background": background or "", "output_format": output_format or "",
                      "ref_count": len(ref_saved)}
    rec = {"task_id": task_id, "status": "pending", "error": "", "filename": "",
           "img_size": "", "elapsed": 0, "created": now, "updated": now,
           "vendor_task_id": "", "vendor_status": "", "params": params_persist}
    with TASKS_LOCK:
        TASKS[task_id] = rec
        try:
            _save_task(rec)
        except Exception as e:
            gen.log(f"[web:{task_id[:8]}] 初始任务记录持久化失败（不影响生成）：{e}")

    run_params = dict(params_persist)      # 已含完整 prompt
    run_params["ref_saved"] = ref_saved
    run_params["ref_max_edge"] = ref_max_edge
    threading.Thread(target=_run_task, args=(task_id, run_params), daemon=True).start()
    return jsonify({"task_id": task_id})


@app.route("/api/task/<task_id>")
def task_status(task_id):
    with TASKS_LOCK:
        t = TASKS.get(task_id)
        if not t:
            # 带 status 字段，前端可据此区分「任务不存在」与网络错误
            return jsonify({"error": "任务不存在（本地任务记录丢失）", "status": "missing"}), 404
        return jsonify(dict(t))


@app.route("/api/history")
def history():
    """历史记录：列出所有已持久化任务（含完整 prompt 与全部设置），按创建时间倒序。

    前端据此展示历史列表，并支持一键把某条记录的提示词/设置回填到表单复用。
    数据源为内存 TASKS（启动时已从 web_outputs/tasks/*.json 全量载入，磁盘为权威）。
    """
    with TASKS_LOCK:
        recs = [dict(r) for r in TASKS.values()]
    recs.sort(key=lambda r: r.get("created") or 0, reverse=True)
    return jsonify({"items": recs[:200]})


@app.route("/output/<path:fname>")
def output(fname):
    return send_from_directory(str(OUT_DIR), fname)


# 启动即恢复未完成任务（在 Flask 起服务前挂好后台线程）
_resume_tasks()


def main():
    ap = argparse.ArgumentParser(description="image2 Web 界面")
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    print(f"[image2 web] 打开浏览器访问  http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
