"""
Material Controller - handles standalone material image generation
"""
from flask import Blueprint, request, current_app, send_file
from models import db, Project, Material, Task
from utils import success_response, error_response, not_found, bad_request
from services import FileService
from services.ai_service_manager import get_ai_service
from services.task_manager import task_manager, generate_material_image_task
from pathlib import Path
from werkzeug.utils import secure_filename
from typing import Optional
import tempfile
import shutil
import time
import zipfile
import io
import base64
import logging

logger = logging.getLogger(__name__)

material_bp = Blueprint('materials', __name__, url_prefix='/api/projects')
material_global_bp = Blueprint('materials_global', __name__, url_prefix='/api/materials')

ALLOWED_MATERIAL_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg'}
ALLOWED_ASPECT_RATIOS = frozenset({'16:9', '21:9', '4:3', '3:2', '5:4', '1:1', '4:5', '2:3', '3:4', '9:16'})


def _generate_image_caption(filepath: str) -> str:
    """Generate AI caption for an uploaded image. Returns empty string on failure."""
    if filepath.lower().endswith('.svg'):
        return ""
    try:
        from PIL import Image

        image = Image.open(filepath)
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

        output_lang = current_app.config.get('OUTPUT_LANGUAGE', 'zh')
        if output_lang == 'en':
            prompt = "Please provide a short description of the main content of this image. Return only the description text without any other explanation."
        else:
            prompt = "请用一句简短的中文描述这张图片的主要内容。只返回描述文字，不要其他解释。"

        provider_format = (current_app.config.get('AI_PROVIDER_FORMAT') or 'gemini').lower()
        caption_model = current_app.config.get('IMAGE_CAPTION_MODEL', 'gemini-3-flash-preview')

        if provider_format == 'openai':
            from openai import OpenAI
            api_key = current_app.config.get('OPENAI_API_KEY', '')
            if not api_key:
                return ""
            client = OpenAI(
                api_key=api_key,
                base_url=current_app.config.get('OPENAI_API_BASE') or None
            )

            buffered = io.BytesIO()
            if image.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                image = background
            image.save(buffered, format="JPEG", quality=95)
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            response = client.chat.completions.create(
                model=caption_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        else:
            # Gemini (default)
            from google import genai
            from google.genai import types
            api_key = current_app.config.get('GOOGLE_API_KEY', '')
            if not api_key:
                return ""
            api_base = current_app.config.get('GOOGLE_API_BASE', '')
            client = genai.Client(
                http_options=types.HttpOptions(base_url=api_base) if api_base else None,
                api_key=api_key
            )
            result = client.models.generate_content(
                model=caption_model,
                contents=[image, prompt],
                config=types.GenerateContentConfig(temperature=0.3)
            )
            return result.text.strip()
    except Exception as e:
        logger.warning(f"Failed to generate caption for {filepath}: {e}")
        return ""


def _build_material_query(filter_project_id: str):
    """Build common material query with project validation."""
    query = Material.query

    if filter_project_id == 'all':
        return query, None
    if filter_project_id == 'none':
        return query.filter(Material.project_id.is_(None)), None

    project = Project.query.get(filter_project_id)
    if not project:
        return None, not_found('Project')

    return query.filter(Material.project_id == filter_project_id), None


def _get_materials_list(filter_project_id: str):
    """
    Common logic to get materials list.
    Returns (materials_list, error_response)
    """
    query, error = _build_material_query(filter_project_id)
    if error:
        return None, error
    
    materials = query.order_by(Material.created_at.desc()).all()
    materials_list = [material.to_dict() for material in materials]
    
    return materials_list, None


def _handle_material_upload(default_project_id: Optional[str] = None):
    """
    Common logic to handle material upload.
    Returns Flask response object.
    """
    try:
        raw_project_id = request.args.get('project_id', default_project_id)
        target_project_id, error = _resolve_target_project_id(raw_project_id)
        if error:
            return error

        file = request.files.get('file')
        material, error = _save_material_file(file, target_project_id)
        if error:
            return error

        result = material.to_dict()

        # Generate AI caption if requested
        generate_caption = request.args.get('generate_caption', '').lower() in ('true', '1', 'yes')
        if generate_caption:
            file_service = FileService(current_app.config['UPLOAD_FOLDER'])
            filepath = file_service.get_absolute_path(material.relative_path)
            caption = _generate_image_caption(filepath)
            result['caption'] = caption

        return success_response(result, status_code=201)

    except Exception as e:
        db.session.rollback()
        return error_response('SERVER_ERROR', str(e), 500)


def _resolve_target_project_id(raw_project_id: Optional[str], allow_none: bool = True):
    """
    Normalize project_id from request.
    Returns (project_id | None, error_response | None)
    """
    if allow_none and (raw_project_id is None or raw_project_id == 'none'):
        return None, None

    if raw_project_id == 'all':
        return None, bad_request("project_id cannot be 'all' when uploading materials")

    if raw_project_id:
        project = Project.query.get(raw_project_id)
        if not project:
            return None, not_found('Project')

    return raw_project_id, None


def _save_material_file(file, target_project_id: Optional[str]):
    """Shared logic for saving uploaded material files to disk and DB."""
    if not file or not file.filename:
        return None, bad_request("file is required")

    filename = secure_filename(file.filename)
    file_ext = Path(filename).suffix.lower()
    if file_ext not in ALLOWED_MATERIAL_EXTENSIONS:
        return None, bad_request(f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_MATERIAL_EXTENSIONS))}")

    file_service = FileService(current_app.config['UPLOAD_FOLDER'])
    if target_project_id:
        materials_dir = file_service.upload_folder / file_service._get_materials_dir(target_project_id)
    else:
        materials_dir = file_service.upload_folder / "materials"
    materials_dir.mkdir(exist_ok=True, parents=True)

    timestamp = int(time.time() * 1000)
    base_name = Path(filename).stem
    unique_filename = f"{base_name}_{timestamp}{file_ext}"

    filepath = materials_dir / unique_filename
    file.save(str(filepath))

    relative_path = str(filepath.relative_to(file_service.upload_folder))
    if target_project_id:
        image_url = file_service.get_file_url(target_project_id, 'materials', unique_filename)
    else:
        image_url = f"/files/materials/{unique_filename}"

    material = Material(
        project_id=target_project_id,
        filename=unique_filename,
        relative_path=relative_path,
        url=image_url
    )

    try:
        db.session.add(material)
        db.session.commit()
        return material, None
    except Exception:
        db.session.rollback()
        raise


@material_bp.route('/<project_id>/materials/generate', methods=['POST'])
def generate_material_image(project_id):
    """
    POST /api/projects/{project_id}/materials/generate - Generate a standalone material image

    Supports multipart/form-data:
    - prompt: Text-to-image prompt (passed directly to the model without modification)
    - ref_image: Main reference image (optional)
    - extra_images: Additional reference images (multiple files, optional)
    
    Note: project_id can be 'none' to generate global materials (not associated with any project)
    """
    try:
        # 支持 'none' 作为特殊值，表示生成全局素材
        if project_id != 'none':
            project = Project.query.get(project_id)
            if not project:
                return not_found('Project')
        else:
            project = None
            project_id = None  # 设置为None表示全局素材

        # Parse request data (prioritize multipart for file uploads)
        if request.is_json:
            data = request.get_json() or {}
            prompt = data.get('prompt', '').strip()
            ref_file = None
            extra_files = []
        else:
            data = request.form.to_dict()
            prompt = (data.get('prompt') or '').strip()
            ref_file = request.files.get('ref_image')
            extra_files = request.files.getlist('extra_images') or []

        aspect_ratio = (data.get('aspect_ratio') or '').strip() or None
        if aspect_ratio and aspect_ratio not in ALLOWED_ASPECT_RATIOS:
            return bad_request(f"Invalid aspect ratio. Allowed values: {', '.join(sorted(ALLOWED_ASPECT_RATIOS))}")

        if not prompt:
            return bad_request("prompt is required")

        # 处理project_id：对于全局素材，使用'global'作为Task的project_id
        # Task模型要求project_id不能为null，但Material可以
        task_project_id = project_id if project_id is not None else 'global'
        
        # 验证project_id（如果不是'global'）
        if task_project_id != 'global':
            project = Project.query.get(task_project_id)
            if not project:
                return not_found('Project')

        # Initialize services
        ai_service = get_ai_service()
        file_service = FileService(current_app.config['UPLOAD_FOLDER'])

        # 创建临时目录保存参考图片（后台任务会清理）
        temp_dir = Path(tempfile.mkdtemp(dir=current_app.config['UPLOAD_FOLDER']))
        temp_dir_str = str(temp_dir)

        try:
            ref_path = None
            # Save main reference image to temp directory if provided
            if ref_file and ref_file.filename:
                ref_filename = secure_filename(ref_file.filename or 'ref.png')
                ref_path = temp_dir / ref_filename
                ref_file.save(str(ref_path))
                ref_path_str = str(ref_path)
            else:
                ref_path_str = None

            # Save additional reference images to temp directory
            additional_ref_images = []
            for extra in extra_files:
                if not extra or not extra.filename:
                    continue
                extra_filename = secure_filename(extra.filename)
                extra_path = temp_dir / extra_filename
                extra.save(str(extra_path))
                additional_ref_images.append(str(extra_path))

            # Create async task for material generation
            task = Task(
                project_id=task_project_id,
                task_type='GENERATE_MATERIAL',
                status='PENDING'
            )
            task.set_progress({
                'total': 1,
                'completed': 0,
                'failed': 0
            })
            db.session.add(task)
            db.session.commit()

            # Get app instance for background task
            app = current_app._get_current_object()

            # Submit background task
            task_manager.submit_task(
                task.id,
                generate_material_image_task,
                task_project_id,  # 传递给任务函数，它会处理'global'的情况
                prompt,
                ai_service,
                file_service,
                ref_path_str,
                additional_ref_images if additional_ref_images else None,
                aspect_ratio or (project.image_aspect_ratio if project else None) or current_app.config.get('DEFAULT_ASPECT_RATIO', '16:9'),
                current_app.config['DEFAULT_RESOLUTION'],
                temp_dir_str,
                app
            )

            # Return task_id immediately (不再清理temp_dir，由后台任务清理)
            return success_response({
                'task_id': task.id,
                'status': 'PENDING'
            }, status_code=202)
        
        except Exception as e:
            # Clean up temp directory on error
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    except Exception as e:
        db.session.rollback()
        return error_response('AI_SERVICE_ERROR', str(e), 503)


@material_bp.route('/<project_id>/materials/generate-ad', methods=['POST'])
def generate_ad_material_image(project_id):
    """
    POST /api/projects/{project_id}/materials/generate-ad - Generate an e-commerce ad image as material.

    Request JSON body:
    {
        "product": { "name": "...", "selling_points": [...], "price": "...", "target_audience": "..." },
        "style":   { "visual_style": "...", "scene": "...", "mood": "..." },
        "layout":  { "aspect_ratio": "1:1", "composition": "...", "text_density": "medium" },
        "tone":    "..."
    }
    Note: project_id can be 'none' to generate global materials.
    """
    try:
        # 支持 'none' 作为特殊值
        if project_id != 'none':
            project = Project.query.get(project_id)
            if not project:
                return not_found('Project')
        else:
            project = None
            project_id = None

        # 兼容两种提交方式：
        # 1. JSON body（无商品图片）
        # 2. multipart/form-data（有商品图片，config 字段为 JSON 字符串）
        import json as _json
        ref_file = None
        extra_files = []
        style_ref_files = []

        if request.is_json:
            body = request.get_json() or {}
            ad_config = body.get('config') or body
        else:
            # multipart
            config_str = request.form.get('config', '{}')
            try:
                body = _json.loads(config_str)
            except Exception:
                return bad_request("config must be valid JSON")
            ad_config = body.get('config') or body
            # 解析商品图片
            ref_file = request.files.get('ref_image')
            extra_files = request.files.getlist('extra_images') or []
            # 解析风格参考图（独立字段，与商品图分开）
            style_ref_files = request.files.getlist('style_ref_images') or []

        if not isinstance(ad_config, dict):
            return bad_request("config must be an object")

        product = ad_config.get('product') or {}
        if not isinstance(product, dict) or not (product.get('name') or '').strip():
            return bad_request("product.name is required")

        # 解析语言与布局参数
        language = body.get('language')
        layout = ad_config.get('layout') or {}
        aspect_ratio = layout.get('aspect_ratio') or current_app.config.get('DEFAULT_ASPECT_RATIO')
        resolution = layout.get('resolution') or current_app.config.get('DEFAULT_RESOLUTION')

        # 处理 Task 的 project_id：全局素材使用 'global'
        task_project_id = project_id if project_id is not None else 'global'
        if task_project_id != 'global':
            project = Project.query.get(task_project_id)
            if not project:
                return not_found('Project')

        ai_service = get_ai_service()
        file_service = FileService(current_app.config['UPLOAD_FOLDER'])

        # 创建临时目录（保持与普通素材接口相同的生命周期）
        temp_dir = Path(tempfile.mkdtemp(dir=current_app.config['UPLOAD_FOLDER']))
        temp_dir_str = str(temp_dir)

        # 保存商品参考图到临时目录
        ref_path_str = None
        additional_ref_list = []

        # 统计商品图数量（用于 prompt 中的位置说明）
        num_product_images = 0
        if ref_file and ref_file.filename:
            ref_filename = secure_filename(ref_file.filename) or 'product_ref.jpg'
            ref_save = temp_dir / ref_filename
            ref_file.save(str(ref_save))
            ref_path_str = str(ref_save)
            num_product_images += 1
        for i, ef in enumerate(extra_files or []):
            if ef and ef.filename:
                ef_name = secure_filename(ef.filename) or f'extra_{i}.jpg'
                ef_save = temp_dir / ef_name
                ef.save(str(ef_save))
                additional_ref_list.append(str(ef_save))
                num_product_images += 1

        # 保存风格参考图（追加在商品图之后）
        has_style_ref_images = False
        for i, srf in enumerate(style_ref_files or []):
            if srf and srf.filename:
                srf_name = secure_filename(srf.filename) or f'style_ref_{i}.jpg'
                srf_save = temp_dir / srf_name
                srf.save(str(srf_save))
                additional_ref_list.append(str(srf_save))
                has_style_ref_images = True

        # 将图片计数与风格参考标志注入 ad_config，供 prompt 函数使用
        ad_config = dict(ad_config)
        ad_config['num_product_images'] = num_product_images
        ad_config['has_style_ref_images'] = has_style_ref_images

        # 根据结构化配置生成专用 prompt
        prompt = ai_service.generate_ad_image_prompt(ad_config, language=language)

        additional_ref_list = additional_ref_list if additional_ref_list else None

        # 创建异步任务
        task = Task(
            project_id=task_project_id,
            task_type='GENERATE_MATERIAL',
            status='PENDING'
        )
        task.set_progress({
            'total': 1,
            'completed': 0,
            'failed': 0
        })
        db.session.add(task)
        db.session.commit()

        app = current_app._get_current_object()

        task_manager.submit_task(
            task.id,
            generate_material_image_task,
            task_project_id,
            prompt,
            ai_service,
            file_service,
            ref_path_str,            # 主商品参考图（可为 None）
            additional_ref_list,     # 额外参考图（可为 None）
            aspect_ratio,
            resolution,
            temp_dir_str,
            app
        )

        return success_response({
            'task_id': task.id,
            'status': 'PENDING'
        }, status_code=202)

    except Exception as e:
        db.session.rollback()
        try:
            if 'temp_dir' in locals() and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        return error_response('AI_SERVICE_ERROR', str(e), 503)


@material_bp.route('/<project_id>/materials/generate-poster', methods=['POST'])
def generate_poster_image(project_id):
    """
    POST /api/projects/{project_id}/materials/generate-poster - Generate a poster image.

    Request (multipart/form-data):
        config (JSON string): { theme, style, extra_description, layout: { aspect_ratio, resolution }, model }
        ref_images (optional files): Reference images for the poster
    """
    try:
        import json as _json

        # Parse project
        if project_id != 'none':
            project = Project.query.get(project_id)
            if not project:
                return not_found('Project')
        else:
            project = None
            project_id = None

        # Parse config from form data
        config_str = request.form.get('config', '{}')
        try:
            config = _json.loads(config_str)
        except Exception:
            return bad_request('config must be valid JSON')

        theme = (config.get('theme') or '').strip()
        if not theme:
            return bad_request('theme is required')

        layout = config.get('layout') or {}
        aspect_ratio = layout.get('aspect_ratio') or current_app.config.get('DEFAULT_ASPECT_RATIO', '16:9')
        resolution = layout.get('resolution') or current_app.config.get('DEFAULT_RESOLUTION', '2K')
        model_name = (config.get('model') or '').strip()

        task_project_id = project_id if project_id is not None else 'global'

        # Generate prompt
        from services.prompts import get_poster_prompt
        prompt = get_poster_prompt(config)

        # Get AI service
        ai_service = get_ai_service()
        file_service = FileService(current_app.config['UPLOAD_FOLDER'])

        # Save reference images to temp dir
        temp_dir = Path(tempfile.mkdtemp(dir=current_app.config['UPLOAD_FOLDER']))
        temp_dir_str = str(temp_dir)
        ref_path_str = None
        additional_ref_list = []

        ref_files = request.files.getlist('ref_images') or []
        for i, rf in enumerate(ref_files):
            if rf and rf.filename:
                rf_name = secure_filename(rf.filename) or f'ref_{i}.jpg'
                rf_save = temp_dir / rf_name
                rf.save(str(rf_save))
                if i == 0:
                    ref_path_str = str(rf_save)
                else:
                    additional_ref_list.append(str(rf_save))

        # Create async task
        task = Task(
            project_id=task_project_id,
            task_type='GENERATE_MATERIAL',
            status='PENDING'
        )
        task.set_progress({'total': 1, 'completed': 0, 'failed': 0})
        db.session.add(task)
        db.session.commit()

        app = current_app._get_current_object()

        # If a specific model is requested (e.g. gpt-image-2), use generate_image_with_model
        # Otherwise use default provider via generate_material_image_task
        if model_name and model_name != 'gemini':
            # Custom model path: submit a poster-specific task
            task_manager.submit_task(
                task.id,
                _generate_poster_with_model_task,
                task_project_id,
                prompt,
                model_name,
                ai_service,
                file_service,
                ref_path_str,
                additional_ref_list if additional_ref_list else None,
                aspect_ratio,
                resolution,
                temp_dir_str,
                app
            )
        else:
            # Default model: reuse existing material generation task
            task_manager.submit_task(
                task.id,
                generate_material_image_task,
                task_project_id,
                prompt,
                ai_service,
                file_service,
                ref_path_str,
                additional_ref_list if additional_ref_list else None,
                aspect_ratio,
                resolution,
                temp_dir_str,
                app
            )

        return success_response({'task_id': task.id, 'status': 'PENDING'}, status_code=202)

    except Exception as e:
        db.session.rollback()
        try:
            if 'temp_dir' in locals() and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        return error_response('AI_SERVICE_ERROR', str(e), 503)


@material_bp.route('/<project_id>/materials/generate-free', methods=['POST'])
def generate_free_image(project_id):
    """
    POST /api/projects/{project_id}/materials/generate-free
    自由生图模式：用户 prompt 原样传给模型，不附加任何 system prompt。
    其他设置（模型/比例/分辨率/参考图）与海报模式一致。

    Request (multipart/form-data):
        config (JSON): { prompt, layout: { aspect_ratio, resolution }, model }
        ref_images (optional files): 参考图
    """
    try:
        import json as _json

        if project_id != 'none':
            project = Project.query.get(project_id)
            if not project:
                return not_found('Project')
        else:
            project = None
            project_id = None

        config_str = request.form.get('config', '{}')
        try:
            config = _json.loads(config_str)
        except Exception:
            return bad_request('config must be valid JSON')

        prompt = (config.get('prompt') or '').strip()
        if not prompt:
            return bad_request('prompt is required')

        layout = config.get('layout') or {}
        aspect_ratio = layout.get('aspect_ratio') or current_app.config.get('DEFAULT_ASPECT_RATIO', '16:9')
        resolution = layout.get('resolution') or current_app.config.get('DEFAULT_RESOLUTION', '2K')
        model_name = (config.get('model') or '').strip()

        task_project_id = project_id if project_id is not None else 'global'

        ai_service = get_ai_service()
        file_service = FileService(current_app.config['UPLOAD_FOLDER'])

        # 保存参考图到临时目录
        temp_dir = Path(tempfile.mkdtemp(dir=current_app.config['UPLOAD_FOLDER']))
        temp_dir_str = str(temp_dir)
        ref_path_str = None
        additional_ref_list = []

        ref_files = request.files.getlist('ref_images') or []
        for i, rf in enumerate(ref_files):
            if rf and rf.filename:
                rf_name = secure_filename(rf.filename) or f'ref_{i}.jpg'
                rf_save = temp_dir / rf_name
                rf.save(str(rf_save))
                if i == 0:
                    ref_path_str = str(rf_save)
                else:
                    additional_ref_list.append(str(rf_save))

        task = Task(
            project_id=task_project_id,
            task_type='GENERATE_MATERIAL',
            status='PENDING'
        )
        task.set_progress({'total': 1, 'completed': 0, 'failed': 0})
        db.session.add(task)
        db.session.commit()

        app = current_app._get_current_object()

        # 复用海报任务函数（它接受任意 prompt，不再附加 system prompt）
        if model_name and model_name != 'gemini':
            task_manager.submit_task(
                task.id,
                _generate_poster_with_model_task,
                task_project_id,
                prompt,
                model_name,
                ai_service,
                file_service,
                ref_path_str,
                additional_ref_list if additional_ref_list else None,
                aspect_ratio,
                resolution,
                temp_dir_str,
                app
            )
        else:
            task_manager.submit_task(
                task.id,
                generate_material_image_task,
                task_project_id,
                prompt,
                ai_service,
                file_service,
                ref_path_str,
                additional_ref_list if additional_ref_list else None,
                aspect_ratio,
                resolution,
                temp_dir_str,
                app
            )

        return success_response({'task_id': task.id, 'status': 'PENDING'}, status_code=202)

    except Exception as e:
        db.session.rollback()
        try:
            if 'temp_dir' in locals() and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        return error_response('AI_SERVICE_ERROR', str(e), 503)


def _aspect_ratio_to_gpt_image_size(aspect_ratio: str, resolution: str = '2K') -> str:
    """
    Map (aspect_ratio, resolution) to a gpt-image-2 supported size string.

    gpt-image-2 size constraints:
      - Max edge length ≤ 3840px
      - Both edges must be multiples of 16
      - Long:short ratio ≤ 3:1
      - Total pixels in [655,360, 8,294,400]

    Tiers (long edge target):
      1K → 1536, 2K → 2048, 4K → 3840 (auto-downscaled when exceeding pixel cap)
    """
    res = (resolution or '2K').upper()

    # 4K tier: long edge up to 3840, but some ratios get capped by total-pixel limit
    mapping_4k = {
        '1:1':  '2880x2880',   # 8,294,400 ≤ cap
        '16:9': '3840x2160',   # 8,294,400 = cap
        '9:16': '2160x3840',
        '4:3':  '2880x2160',
        '3:4':  '2160x2880',
        '3:2':  '3072x2048',
        '2:3':  '2048x3072',
        '21:9': '3840x1648',
        '5:4':  '3072x2448',
    }
    mapping_2k = {
        '1:1':  '2048x2048',
        '16:9': '2048x1152',
        '9:16': '1152x2048',
        '4:3':  '2048x1536',
        '3:4':  '1536x2048',
        '3:2':  '2048x1360',
        '2:3':  '1360x2048',
        '21:9': '2048x880',
        '5:4':  '2048x1632',
    }
    mapping_1k = {
        '1:1':  '1024x1024',
        '16:9': '1536x1024',  # not exact 16:9 but closest 1K landscape
        '9:16': '1024x1536',
        '4:3':  '1536x1152',
        '3:4':  '1152x1536',
        '3:2':  '1536x1024',
        '2:3':  '1024x1536',
        '21:9': '1536x656',
        '5:4':  '1280x1024',
    }

    if res == '4K':
        return mapping_4k.get(aspect_ratio, '3840x2160')
    if res == '1K':
        return mapping_1k.get(aspect_ratio, '1536x1024')
    # default 2K
    return mapping_2k.get(aspect_ratio, '2048x1152')


def _direct_call_gpt_image(prompt: str, model_name: str, api_key: str, api_base: str,
                            aspect_ratio: str, resolution: str, ref_paths: list,
                            timeout: float = 600.0):
    """
    Direct HTTP call to OpenAI-compatible images API (e.g. AiHubMix /v1/images/generations).
    Bypasses OpenAI SDK to give us full visibility into request/response for debugging.
    Returns PIL.Image on success, raises Exception with full server response body on failure.
    """
    import requests
    import base64 as _b64
    from io import BytesIO as _BIO
    from PIL import Image as _PIL

    # Normalize api_base: strip trailing slash and any '/v1' or '/gemini' suffix duplication
    base = (api_base or 'https://aihubmix.com/v1').rstrip('/')
    # If base ends with /gemini (Google route), force switch to /v1 for OpenAI-style endpoint
    if base.endswith('/gemini'):
        base = base[: -len('/gemini')] + '/v1'
    if not base.endswith('/v1'):
        # If a custom proxy is given without /v1, append it to be safe
        if '/v1' not in base:
            base = base + '/v1'

    size = _aspect_ratio_to_gpt_image_size(aspect_ratio, resolution)
    quality = 'high' if str(resolution).upper() in ('2K', '4K') else 'auto'

    headers = {
        'Authorization': f'Bearer {api_key}',
    }

    masked_key = (api_key[:8] + '...' + api_key[-4:]) if api_key and len(api_key) > 12 else '***'
    logger.info(f'[gpt-image direct] model={model_name}, base={base}, key={masked_key}, size={size}, quality={quality}, ref_count={len(ref_paths) if ref_paths else 0}')

    # Bypass system HTTP_PROXY / HTTPS_PROXY env vars: aihubmix.com is directly accessible
    # from CN networks; routing through a local proxy (e.g. V2Ray) often gets dropped.
    no_proxy = {'http': None, 'https': None}
    session = requests.Session()
    session.trust_env = False  # ignore HTTP_PROXY / HTTPS_PROXY / NO_PROXY env

    if ref_paths:
        # Use /images/edits with multipart
        url = f'{base}/images/edits'
        files = []
        for i, p in enumerate(ref_paths):
            files.append(('image[]', (f'ref_{i}.png', open(p, 'rb'), 'image/png')))
        data = {
            'model': model_name,
            'prompt': prompt,
            'n': '1',
            'size': size,
        }
        try:
            resp = session.post(url, headers=headers, data=data, files=files, timeout=timeout, proxies=no_proxy)
        finally:
            for _, fobj in files:
                try:
                    fobj[1].close()
                except Exception:
                    pass
    else:
        url = f'{base}/images/generations'
        payload = {
            'model': model_name,
            'prompt': prompt,
            'n': 1,
            'size': size,
            'quality': quality,
        }
        headers['Content-Type'] = 'application/json'
        resp = session.post(url, headers=headers, json=payload, timeout=timeout, proxies=no_proxy)

    logger.info(f'[gpt-image direct] response status={resp.status_code}')

    if resp.status_code != 200:
        # Surface the entire server response so the user can see the real cause
        body = resp.text[:2000]
        raise Exception(f'gpt-image API error {resp.status_code}: {body}')

    result = resp.json()
    item = (result.get('data') or [{}])[0]

    if item.get('b64_json'):
        img_bytes = _b64.b64decode(item['b64_json'])
        return _PIL.open(_BIO(img_bytes)).convert('RGB')
    if item.get('url'):
        r2 = requests.get(item['url'], timeout=120)
        r2.raise_for_status()
        return _PIL.open(_BIO(r2.content)).convert('RGB')

    raise Exception(f'gpt-image API returned no image data: {str(result)[:500]}')


def _generate_poster_with_model_task(
    task_id, project_id, prompt, model_name, ai_service, file_service,
    ref_path_str, additional_ref_list, aspect_ratio, resolution,
    temp_dir_str, app
):
    """Background task: generate poster image with a specific model."""
    import os
    with app.app_context():
        try:
            # Collect all ref image paths
            ref_paths = []
            if ref_path_str:
                ref_paths.append(ref_path_str)
            if additional_ref_list:
                ref_paths.extend(additional_ref_list)

            model_lower = (model_name or '').lower()
            is_gpt_image = model_lower.startswith('gpt-image') or model_lower.startswith('dall-e')

            if is_gpt_image:
                # Helper: filter out invalid placeholders / empty values
                def _valid(v):
                    if not v:
                        return False
                    s = str(v).strip()
                    if not s:
                        return False
                    bad = {'your-api-key-here', 'your_api_key_here', 'sk-xxx', 'changeme'}
                    if s.lower() in bad or s.lower().startswith('your-api-key'):
                        return False
                    return True

                # Pick the first valid key. AiHubMix uses one key for both Gemini & OpenAI routes,
                # so GOOGLE_API_KEY (the real key in .env) works for /v1/images/generations as well.
                api_key = ''
                for cand in (
                    os.getenv('OPENAI_API_KEY'),
                    current_app.config.get('OPENAI_API_KEY'),
                    os.getenv('GOOGLE_API_KEY'),
                    current_app.config.get('GOOGLE_API_KEY'),
                ):
                    if _valid(cand):
                        api_key = cand
                        break

                # Force AiHubMix /v1 endpoint for gpt-image regardless of OPENAI_API_BASE,
                # because the env may point to api.openai.com which has no AiHubMix proxy.
                env_base = os.getenv('OPENAI_API_BASE', '').strip()
                if env_base and 'aihubmix' in env_base.lower():
                    api_base = env_base
                else:
                    api_base = 'https://aihubmix.com/v1'

                if not api_key:
                    raise Exception('No valid API key found for gpt-image (OPENAI_API_KEY / GOOGLE_API_KEY)')

                from config import get_config
                timeout = float(getattr(get_config(), 'OPENAI_TIMEOUT', 600.0) or 600.0)
                # gpt-image-2 generation can take 5-10 minutes; enforce a minimum 600s
                if timeout < 600.0:
                    timeout = 600.0

                image = _direct_call_gpt_image(
                    prompt=prompt,
                    model_name=model_name,
                    api_key=api_key,
                    api_base=api_base,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    ref_paths=ref_paths,
                    timeout=timeout,
                )
            else:
                image = ai_service.generate_image_with_model(
                    prompt=prompt,
                    model_name=model_name,
                    ref_image_paths=ref_paths if ref_paths else None,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                )

            if image is None:
                raise Exception('Image generation returned None')

            # Save the generated image as a material (mirror generate_material_image_task)
            from pathlib import Path as _Path
            from datetime import datetime as _dt
            actual_project_id = None if (project_id == 'global' or project_id is None) else project_id
            relative_path = file_service.save_material_image(image, actual_project_id)
            filename = _Path(relative_path).name
            image_url = file_service.get_file_url(actual_project_id, 'materials', filename)

            material = Material(
                project_id=actual_project_id,
                filename=filename,
                relative_path=relative_path,
                url=image_url,
            )
            db.session.add(material)

            # Mark task as completed
            task = Task.query.get(task_id)
            if task:
                task.status = 'COMPLETED'
                task.completed_at = _dt.utcnow()
                task.set_progress({
                    'total': 1,
                    'completed': 1,
                    'failed': 0,
                    'material_id': material.id,
                    'image_url': image_url,
                })
            db.session.commit()
            logger.info(f'✅ Poster Task {task_id} COMPLETED - Material {material.id} generated via {model_name}')
            return material
        finally:
            # Clean up temp directory
            try:
                if temp_dir_str:
                    shutil.rmtree(temp_dir_str, ignore_errors=True)
            except Exception:
                pass


@material_bp.route('/<project_id>/materials', methods=['GET'])
def list_materials(project_id):
    """
    GET /api/projects/{project_id}/materials - List materials for a specific project
    
    Returns:
        List of material images with filename, url, and metadata for the specified project
    """
    try:
        materials_list, error = _get_materials_list(project_id)
        if error:
            return error
        
        return success_response({
            "materials": materials_list,
            "count": len(materials_list)
        })
    
    except Exception as e:
        return error_response('SERVER_ERROR', str(e), 500)


@material_bp.route('/<project_id>/materials/upload', methods=['POST'])
def upload_material(project_id):
    """
    POST /api/projects/{project_id}/materials/upload - Upload a material image
    
    Supports multipart/form-data:
    - file: Image file (required)
    - project_id: Optional query parameter, defaults to path parameter if not provided
    
    Returns:
        Material info with filename, url, and metadata
    """
    return _handle_material_upload(default_project_id=project_id)


@material_global_bp.route('', methods=['GET'])
def list_all_materials():
    """
    GET /api/materials - Global materials endpoint for complex queries
    
    Query params:
        - project_id: Filter by project_id
          * 'all' (default): Get all materials regardless of project
          * 'none': Get only materials without a project (global materials)
          * <project_id>: Get materials for specific project
    
    Returns:
        List of material images with filename, url, and metadata
    """
    try:
        filter_project_id = request.args.get('project_id', 'all')
        materials_list, error = _get_materials_list(filter_project_id)
        if error:
            return error
        
        return success_response({
            "materials": materials_list,
            "count": len(materials_list)
        })
    
    except Exception as e:
        return error_response('SERVER_ERROR', str(e), 500)


@material_global_bp.route('/upload', methods=['POST'])
def upload_material_global():
    """
    POST /api/materials/upload - Upload a material image (global, not bound to a project)
    
    Supports multipart/form-data:
    - file: Image file (required)
    - project_id: Optional query parameter to associate with a project
    
    Returns:
        Material info with filename, url, and metadata
    """
    return _handle_material_upload(default_project_id=None)


@material_global_bp.route('/<material_id>', methods=['DELETE'])
def delete_material(material_id):
    """
    DELETE /api/materials/{material_id} - Delete a material and its file
    """
    try:
        material = Material.query.get(material_id)
        if not material:
            return not_found('Material')

        file_service = FileService(current_app.config['UPLOAD_FOLDER'])
        material_path = Path(file_service.get_absolute_path(material.relative_path))

        # First, delete the database record to ensure data consistency
        db.session.delete(material)
        db.session.commit()

        # Then, attempt to delete the file. If this fails, log the error
        # but still return a success response. This leaves an orphan file,
        try:
            if material_path.exists():
                material_path.unlink(missing_ok=True)
        except OSError as e:
            current_app.logger.warning(f"Failed to delete file for material {material_id} at {material_path}: {e}")

        return success_response({"id": material_id})
    except Exception as e:
        db.session.rollback()
        return error_response('SERVER_ERROR', str(e), 500)


@material_global_bp.route('/associate', methods=['POST'])
def associate_materials_to_project():
    """
    POST /api/materials/associate - Associate materials to a project by URLs

    Request body (JSON):
    {
        "project_id": "project_id",
        "material_urls": ["url1", "url2", ...]
    }

    Returns:
        List of associated material IDs and count
    """
    try:
        data = request.get_json() or {}
        project_id = data.get('project_id')
        material_urls = data.get('material_urls', [])

        if not project_id:
            return bad_request("project_id is required")

        if not material_urls or not isinstance(material_urls, list):
            return bad_request("material_urls must be a non-empty array")

        # Validate project exists
        project = Project.query.get(project_id)
        if not project:
            return not_found('Project')

        # Find materials by URLs and update their project_id
        updated_ids = []
        materials_to_update = Material.query.filter(
            Material.url.in_(material_urls),
            Material.project_id.is_(None)
        ).all()
        for material in materials_to_update:
            material.project_id = project_id
            updated_ids.append(material.id)

        db.session.commit()

        return success_response({
            "updated_ids": updated_ids,
            "count": len(updated_ids)
        })

    except Exception as e:
        db.session.rollback()
        return error_response('SERVER_ERROR', str(e), 500)


@material_global_bp.route('/download', methods=['POST'])
def download_materials_zip():
    """Bundle requested materials into a ZIP and stream it back."""
    body = request.get_json(silent=True) or {}
    ids = body.get('material_ids')

    if not ids or not isinstance(ids, list):
        return bad_request("material_ids must be a non-empty list")

    MAX_BATCH = 200
    if len(ids) > MAX_BATCH:
        return bad_request(f"Too many materials requested (max {MAX_BATCH})")

    rows = Material.query.filter(Material.id.in_(ids)).all()
    if not rows:
        return not_found('Materials')

    tmp = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024)
    try:
        fs = FileService(current_app.config['UPLOAD_FOLDER'])

        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zf:
            for row in rows:
                abs_path = Path(fs.get_absolute_path(row.relative_path))
                if not abs_path.is_file():
                    current_app.logger.warning("Skipping missing file for material %s", row.id)
                    continue
                zf.write(str(abs_path), row.filename)

        tmp.seek(0)
        fname = f"materials_{int(time.time())}.zip"

        return send_file(tmp, mimetype='application/zip',
                         as_attachment=True, download_name=fname)
    except Exception:
        tmp.close()
        current_app.logger.exception("Failed to build materials zip")
        return error_response('SERVER_ERROR', 'Failed to create zip archive', 500)

