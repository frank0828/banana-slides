"""
AI Service - handles all AI model interactions
Based on demo.py and gemini_genai.py
TODO: use structured output API
"""
import os
import json
import re
import logging
import requests
from typing import List, Dict, Optional, Union
from textwrap import dedent
from PIL import Image
from tenacity import retry, stop_after_attempt, retry_if_exception_type
from .prompts import (
    get_outline_generation_prompt,
    get_outline_parsing_prompt,
    get_page_description_prompt,
    get_image_generation_prompt,
    get_image_edit_prompt,
    get_description_to_outline_prompt,
    get_description_split_prompt,
    get_outline_refinement_prompt,
    get_descriptions_refinement_prompt,
    get_layout_pages_prompt,
    get_ad_image_prompt,
)
from .ai_providers import get_text_provider, get_image_provider, TextProvider, ImageProvider
from config import get_config

logger = logging.getLogger(__name__)


class ProjectContext:
    """项目上下文数据类，统一管理 AI 需要的所有项目信息"""
    
    def __init__(self, project_or_dict, reference_files_content: Optional[List[Dict[str, str]]] = None):
        """
        Args:
            project_or_dict: 项目对象（Project model）或项目字典（project.to_dict()）
            reference_files_content: 参考文件内容列表
        """
        # 支持直接传入 Project 对象，避免 to_dict() 调用，提升性能
        if hasattr(project_or_dict, 'idea_prompt'):
            # 是 Project 对象
            self.idea_prompt = project_or_dict.idea_prompt
            self.outline_text = project_or_dict.outline_text
            self.description_text = project_or_dict.description_text
            self.creation_type = project_or_dict.creation_type or 'idea'
        else:
            # 是字典
            self.idea_prompt = project_or_dict.get('idea_prompt')
            self.outline_text = project_or_dict.get('outline_text')
            self.description_text = project_or_dict.get('description_text')
            self.creation_type = project_or_dict.get('creation_type', 'idea')
        
        self.reference_files_content = reference_files_content or []
    
    def to_dict(self) -> Dict:
        """转换为字典，方便传递"""
        return {
            'idea_prompt': self.idea_prompt,
            'outline_text': self.outline_text,
            'description_text': self.description_text,
            'creation_type': self.creation_type,
            'reference_files_content': self.reference_files_content
        }


class AIService:
    """Service for AI model interactions using pluggable providers"""
    
    def __init__(self, text_provider: TextProvider = None, image_provider: ImageProvider = None):
        """
        Initialize AI service with providers
        
        Args:
            text_provider: Optional pre-configured TextProvider. If None, created from factory.
            image_provider: Optional pre-configured ImageProvider. If None, created from factory.
        """
        config = get_config()

        # 优先使用 Flask app.config（可由 Settings 覆盖），否则回退到 Config 默认值
        try:
            from flask import current_app, has_app_context
        except ImportError:
            current_app = None  # type: ignore
            has_app_context = lambda: False  # type: ignore

        if has_app_context() and current_app and hasattr(current_app, "config"):
            self.text_model = current_app.config.get("TEXT_MODEL", config.TEXT_MODEL)
            self.image_model = current_app.config.get("IMAGE_MODEL", config.IMAGE_MODEL)
        else:
            self.text_model = config.TEXT_MODEL
            self.image_model = config.IMAGE_MODEL
        
        # Use provided providers or create from factory based on AI_PROVIDER_FORMAT (from Flask config or env var)
        self.text_provider = text_provider or get_text_provider(model=self.text_model)
        self.image_provider = image_provider or get_image_provider(model=self.image_model)
    
    @staticmethod
    def extract_image_urls_from_markdown(text: str) -> List[str]:
        """
        从 markdown 文本中提取图片 URL
        
        Args:
            text: Markdown 文本，可能包含 ![](url) 格式的图片
            
        Returns:
            图片 URL 列表（包括 http/https URL 和 /files/mineru/ 开头的本地路径）
        """
        if not text:
            return []
        
        # 匹配 markdown 图片语法: ![](url) 或 ![alt](url)
        pattern = r'!\[.*?\]\((.*?)\)'
        matches = re.findall(pattern, text)
        
        # 过滤掉空字符串，支持 http/https URL 和 /files/mineru/ 开头的本地路径
        urls = []
        for url in matches:
            url = url.strip()
            if url and (url.startswith('http://') or url.startswith('https://') or url.startswith('/files/mineru/')):
                urls.append(url)
        
        return urls
    
    @staticmethod
    def remove_markdown_images(text: str) -> str:
        """
        从文本中移除 Markdown 图片链接，只保留 alt text（描述文字）
        
        Args:
            text: 包含 Markdown 图片语法的文本
            
        Returns:
            移除图片链接后的文本，保留描述文字
        """
        if not text:
            return text
        
        # 将 ![描述文字](url) 替换为 描述文字
        # 如果没有描述文字（空的 alt text），则完全删除该图片链接
        def replace_image(match):
            alt_text = match.group(1).strip()
            # 如果有描述文字，保留它；否则删除整个链接
            return alt_text if alt_text else ''
        
        pattern = r'!\[(.*?)\]\([^\)]+\)'
        cleaned_text = re.sub(pattern, replace_image, text)
        
        # 清理可能产生的多余空行
        cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
        
        return cleaned_text
    
    @retry(
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((json.JSONDecodeError, ValueError)),
        reraise=True
    )
    def generate_json(self, prompt: str, thinking_budget: int = 1000) -> Union[Dict, List]:
        """
        生成并解析JSON，如果解析失败则重新生成
        
        Args:
            prompt: 生成提示词
            thinking_budget: 思考预算
            
        Returns:
            解析后的JSON对象（字典或列表）
            
        Raises:
            json.JSONDecodeError: JSON解析失败（重试3次后仍失败）
        """
        # 调用AI生成文本
        response_text = self.text_provider.generate_text(prompt, thinking_budget=thinking_budget)
        
        # 清理响应文本：移除markdown代码块标记和多余空白
        cleaned_text = response_text.strip().strip("```json").strip("```").strip()
        
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败，将重新生成。原始文本: {cleaned_text[:200]}... 错误: {str(e)}")
            raise
    
    @staticmethod
    def _convert_mineru_path_to_local(mineru_path: str) -> Optional[str]:
        """
        将 /files/mineru/{extract_id}/{rel_path} 格式的路径转换为本地文件系统路径（支持前缀匹配）
        
        Args:
            mineru_path: MinerU URL 路径，格式为 /files/mineru/{extract_id}/{rel_path}
            
        Returns:
            本地文件系统路径，如果转换失败则返回 None
        """
        from utils.path_utils import find_mineru_file_with_prefix
        
        matched_path = find_mineru_file_with_prefix(mineru_path)
        return str(matched_path) if matched_path else None
    
    @staticmethod
    def download_image_from_url(url: str) -> Optional[Image.Image]:
        """
        从 URL 下载图片并返回 PIL Image 对象
        
        Args:
            url: 图片 URL
            
        Returns:
            PIL Image 对象，如果下载失败则返回 None
        """
        try:
            logger.debug(f"Downloading image from URL: {url}")
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # 从响应内容创建 PIL Image
            image = Image.open(response.raw)
            # 确保图片被加载
            image.load()
            logger.debug(f"Successfully downloaded image: {image.size}, {image.mode}")
            return image
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {str(e)}")
            return None
    
    def generate_outline(self, project_context: ProjectContext, language: str = None) -> List[Dict]:
        """
        Generate PPT outline from idea prompt
        Based on demo.py gen_outline()
        
        Args:
            project_context: 项目上下文对象，包含所有原始信息
            
        Returns:
            List of outline items (may contain parts with pages or direct pages)
        """
        outline_prompt = get_outline_generation_prompt(project_context, language)
        outline = self.generate_json(outline_prompt, thinking_budget=1000)
        return outline
    
    def parse_outline_text(self, project_context: ProjectContext, language: str = None) -> List[Dict]:
        """
        Parse user-provided outline text into structured outline format
        This method analyzes the text and splits it into pages without modifying the original text
        
        Args:
            project_context: 项目上下文对象，包含所有原始信息
        
        Returns:
            List of outline items (may contain parts with pages or direct pages)
        """
        parse_prompt = get_outline_parsing_prompt(project_context, language)
        outline = self.generate_json(parse_prompt, thinking_budget=1000)
        return outline
    
    def flatten_outline(self, outline: List[Dict]) -> List[Dict]:
        """
        Flatten outline structure to page list
        Based on demo.py flatten_outline()
        """
        pages = []
        for item in outline:
            if "part" in item and "pages" in item:
                # This is a part, expand its pages
                for page in item["pages"]:
                    page_with_part = page.copy()
                    page_with_part["part"] = item["part"]
                    pages.append(page_with_part)
            else:
                # This is a direct page
                pages.append(item)
        return pages
    
    def generate_page_description(self, project_context: ProjectContext, outline: List[Dict], 
                                 page_outline: Dict, page_index: int, language='zh') -> str:
        """
        Generate description for a single page
        Based on demo.py gen_desc() logic
        
        Args:
            project_context: 项目上下文对象，包含所有原始信息
            outline: Complete outline
            page_outline: Outline for this specific page
            page_index: Page number (1-indexed)
        
        Returns:
            Text description for the page
        """
        part_info = f"\nThis page belongs to: {page_outline['part']}" if 'part' in page_outline else ""
        
        desc_prompt = get_page_description_prompt(
            project_context=project_context,
            outline=outline,
            page_outline=page_outline,
            page_index=page_index,
            part_info=part_info,
            language=language
        )
        
        response_text = self.text_provider.generate_text(desc_prompt, thinking_budget=1000)
        
        return dedent(response_text)
    
    def generate_outline_text(self, outline: List[Dict]) -> str:
        """
        Convert outline to text format for prompts
        Based on demo.py gen_outline_text()
        """
        text_parts = []
        for i, item in enumerate(outline, 1):
            if "part" in item and "pages" in item:
                text_parts.append(f"{i}. {item['part']}")
            else:
                text_parts.append(f"{i}. {item.get('title', 'Untitled')}")
        result = "\n".join(text_parts)
        return dedent(result)
    
    def generate_image_prompt(self, outline: List[Dict], page: Dict, 
                            page_desc: str, page_index: int, 
                            has_material_images: bool = False,
                            extra_requirements: Optional[str] = None,
                            language='zh') -> str:
        """
        Generate image generation prompt for a page
        Based on demo.py gen_prompts()
        
        Args:
            outline: Complete outline
            page: Page outline data
            page_desc: Page description text
            page_index: Page number (1-indexed)
            has_material_images: 是否有素材图片（从项目描述中提取的图片）
            extra_requirements: Optional extra requirements to apply to all pages
        
        Returns:
            Image generation prompt
        """
        outline_text = self.generate_outline_text(outline)
        
        # Determine current section
        if 'part' in page:
            current_section = page['part']
        else:
            current_section = f"{page.get('title', 'Untitled')}"
        
        # 在传给文生图模型之前，移除 Markdown 图片链接
        # 图片本身已经通过 additional_ref_images 传递，只保留文字描述
        cleaned_page_desc = self.remove_markdown_images(page_desc)
        
        prompt = get_image_generation_prompt(
            page_desc=cleaned_page_desc,
            outline_text=outline_text,
            current_section=current_section,
            has_material_images=has_material_images,
            extra_requirements=extra_requirements,
            language=language
        )
        
        return prompt
    
    def generate_ad_image_prompt(self, ad_config: Dict, language: str = None) -> str:
        """
        生成电商广告图的图片生成 prompt（不影响现有 PPT 流程）
        
        Args:
            ad_config: 包含 product/style/layout/tone 的配置字典
            language: 语言代码（可选）
        
        Returns:
            图片生成 prompt 字符串
        """
        return get_ad_image_prompt(ad_config, language)

    def generate_image(self, prompt: str, ref_image_path: Optional[str] = None, 
                      aspect_ratio: str = "16:9", resolution: str = "2K",
                      additional_ref_images: Optional[List[Union[str, Image.Image]]] = None) -> Optional[Image.Image]:
        """
        Generate image using configured image provider
        Based on gemini_genai.py gen_image()
        
        Args:
            prompt: Image generation prompt
            ref_image_path: Path to reference image (optional). If None, will generate based on prompt only.
            aspect_ratio: Image aspect ratio
            resolution: Image resolution (note: OpenAI format only supports 1K)
            additional_ref_images: 额外的参考图片列表，可以是本地路径、URL 或 PIL Image 对象
        
        Returns:
            PIL Image object or None if failed
        
        Raises:
            Exception with detailed error message if generation fails
        """
        try:
            logger.debug(f"Reference image: {ref_image_path}")
            if additional_ref_images:
                logger.debug(f"Additional reference images: {len(additional_ref_images)}")
            logger.debug(f"Config - aspect_ratio: {aspect_ratio}, resolution: {resolution}")

            # 构建参考图片列表
            ref_images = []
            
            # 添加主参考图片（如果提供了路径）
            if ref_image_path:
                if not os.path.exists(ref_image_path):
                    raise FileNotFoundError(f"Reference image not found: {ref_image_path}")
                main_ref_image = Image.open(ref_image_path)
                ref_images.append(main_ref_image)
            
            # 添加额外的参考图片
            if additional_ref_images:
                for ref_img in additional_ref_images:
                    if isinstance(ref_img, Image.Image):
                        # 已经是 PIL Image 对象
                        ref_images.append(ref_img)
                    elif isinstance(ref_img, str):
                        # 可能是本地路径或 URL
                        if os.path.exists(ref_img):
                            # 本地路径
                            ref_images.append(Image.open(ref_img))
                        elif ref_img.startswith('http://') or ref_img.startswith('https://'):
                            # URL，需要下载
                            downloaded_img = self.download_image_from_url(ref_img)
                            if downloaded_img:
                                ref_images.append(downloaded_img)
                            else:
                                logger.warning(f"Failed to download image from URL: {ref_img}, skipping...")
                        elif ref_img.startswith('/files/mineru/'):
                            # MinerU 本地文件路径，需要转换为文件系统路径（支持前缀匹配）
                            local_path = self._convert_mineru_path_to_local(ref_img)
                            if local_path and os.path.exists(local_path):
                                ref_images.append(Image.open(local_path))
                                logger.debug(f"Loaded MinerU image from local path: {local_path}")
                            else:
                                logger.warning(f"MinerU image file not found (with prefix matching): {ref_img}, skipping...")
                        else:
                            logger.warning(f"Invalid image reference: {ref_img}, skipping...")
            
            logger.debug(f"Calling image provider for generation with {len(ref_images)} reference images...")
            
            # 使用 image_provider 生成图片
            return self.image_provider.generate_image(
                prompt=prompt,
                ref_images=ref_images if ref_images else None,
                aspect_ratio=aspect_ratio,
                resolution=resolution
            )
            
        except Exception as e:
            error_detail = f"Error generating image: {type(e).__name__}: {str(e)}"
            logger.error(error_detail, exc_info=True)
            raise Exception(error_detail) from e
    
    def edit_image(self, prompt: str, current_image_path: str,
                  aspect_ratio: str = "16:9", resolution: str = "2K",
                  original_description: str = None,
                  additional_ref_images: Optional[List[Union[str, Image.Image]]] = None) -> Optional[Image.Image]:
        """
        Edit existing image with natural language instruction
        Uses current image as reference
        
        Args:
            prompt: Edit instruction
            current_image_path: Path to current page image
            aspect_ratio: Image aspect ratio
            resolution: Image resolution
            original_description: Original page description to include in prompt
            additional_ref_images: 额外的参考图片列表，可以是本地路径、URL 或 PIL Image 对象
        
        Returns:
            PIL Image object or None if failed
        """
        # Build edit instruction with original description if available
        edit_instruction = get_image_edit_prompt(
            edit_instruction=prompt,
            original_description=original_description
        )
        return self.generate_image(edit_instruction, current_image_path, aspect_ratio, resolution, additional_ref_images)
    
    def parse_description_to_outline(self, project_context: ProjectContext, language='zh') -> List[Dict]:
        """
        从描述文本解析出大纲结构
        
        Args:
            project_context: 项目上下文对象，包含所有原始信息
        
        Returns:
            List of outline items (may contain parts with pages or direct pages)
        """
        parse_prompt = get_description_to_outline_prompt(project_context, language)
        outline = self.generate_json(parse_prompt, thinking_budget=1000)
        return outline
    
    def parse_description_to_page_descriptions(self, project_context: ProjectContext, 
                                               outline: List[Dict],
                                               language='zh') -> List[str]:
        """
        从描述文本切分出每页描述
        
        Args:
            project_context: 项目上下文对象，包含所有原始信息
            outline: 已解析出的大纲结构
        
        Returns:
            List of page descriptions (strings), one for each page in the outline
        """
        split_prompt = get_description_split_prompt(project_context, outline, language)
        descriptions = self.generate_json(split_prompt, thinking_budget=1000)
        
        # 确保返回的是字符串列表
        if isinstance(descriptions, list):
            return [str(desc) for desc in descriptions]
        else:
            raise ValueError("Expected a list of page descriptions, but got: " + str(type(descriptions)))
    
    def refine_outline(self, current_outline: List[Dict], user_requirement: str,
                      project_context: ProjectContext,
                      previous_requirements: Optional[List[str]] = None,
                      language='zh') -> List[Dict]:
        """
        根据用户要求修改已有大纲
        
        Args:
            current_outline: 当前的大纲结构
            user_requirement: 用户的新要求
            project_context: 项目上下文对象，包含所有原始信息
            previous_requirements: 之前的修改要求列表（可选）
        
        Returns:
            修改后的大纲结构
        """
        refinement_prompt = get_outline_refinement_prompt(
            current_outline=current_outline,
            user_requirement=user_requirement,
            project_context=project_context,
            previous_requirements=previous_requirements,
            language=language
        )
        outline = self.generate_json(refinement_prompt, thinking_budget=1000)
        return outline
    
    def refine_descriptions(self, current_descriptions: List[Dict], user_requirement: str,
                           project_context: ProjectContext,
                           outline: List[Dict] = None,
                           previous_requirements: Optional[List[str]] = None,
                           language='zh') -> List[str]:
        """
        根据用户要求修改已有页面描述
        
        Args:
            current_descriptions: 当前的页面描述列表，每个元素包含 {index, title, description_content}
            user_requirement: 用户的新要求
            project_context: 项目上下文对象，包含所有原始信息
            outline: 完整的大纲结构（可选）
            previous_requirements: 之前的修改要求列表（可选）
        
        Returns:
            修改后的页面描述列表（字符串列表）
        """
        refinement_prompt = get_descriptions_refinement_prompt(
            current_descriptions=current_descriptions,
            user_requirement=user_requirement,
            project_context=project_context,
            outline=outline,
            previous_requirements=previous_requirements,
            language=language
        )
        descriptions = self.generate_json(refinement_prompt, thinking_budget=1000)
        
        # 确保返回的是字符串列表
        if isinstance(descriptions, list):
            return [str(desc) for desc in descriptions]
        else:
            raise ValueError("Expected a list of page descriptions, but got: " + str(type(descriptions)))
    
    def parse_layout_pages(self, project_context: ProjectContext, language='zh') -> Dict:
        """
        直接排版模式：解析用户输入的完整页面内容，切分为各页但不修改文字
        优先使用规则分页（识别 --- 分隔符和 # 标题），没有明确分隔符时才使用AI
        
        Args:
            project_context: 项目上下文对象，包含所有原始信息
        
        Returns:
            Dict with 'pages' list, each containing title, points, and description
        """
        layout_text = project_context.description_text or ""
        
        # 尝试使用规则分页
        rule_based_pages = self._parse_layout_by_rules(layout_text)
        
        if rule_based_pages:
            logger.info(f"使用规则分页成功，识别到 {len(rule_based_pages)} 个页面")
            return {'pages': rule_based_pages}
        
        # 规则分页失败，使用AI
        logger.info("未检测到明确分隔符，使用AI进行分页")
        parse_prompt = get_layout_pages_prompt(project_context, language)
        result = self.generate_json(parse_prompt, thinking_budget=1000)
        
        # 验证返回格式
        if not isinstance(result, dict) or 'pages' not in result:
            raise ValueError("Expected a dict with 'pages' key, but got: " + str(type(result)))
        
        return result
    
    @staticmethod
    def _parse_layout_by_rules(text: str) -> Optional[List[Dict]]:
        """
        使用规则解析直接排版内容，识别 Markdown 分隔符和标题层级
        
        优先级：
        1. --- 分隔符（最高优先级，明确的页面分隔）
        2. # 一级标题（作为页面标题）
        3. ## 二级标题（如果没有一级标题，则使用二级标题）
        
        Args:
            text: 用户输入的文本内容
        
        Returns:
            页面列表，如果无法识别则返回 None
        """
        if not text or not text.strip():
            return None
        
        lines = text.split('\n')
        pages = []
        current_page_lines = []
        current_title = None
        has_explicit_separator = False
        
        # 检查是否有 --- 分隔符
        for line in lines:
            stripped = line.strip()
            if stripped == '---' or stripped == '___' or stripped == '***':
                has_explicit_separator = True
                break
        
        if has_explicit_separator:
            # 使用 --- 分隔符进行分页
            logger.info("检测到 Markdown 分隔符 (---/___/***)，使用分隔符分页")
            for line in lines:
                stripped = line.strip()
                if stripped in ['---', '___', '***']:
                    # 遇到分隔符，保存当前页面
                    if current_page_lines:
                        page = AIService._extract_page_from_lines(current_page_lines, current_title)
                        if page:
                            pages.append(page)
                        current_page_lines = []
                        current_title = None
                else:
                    current_page_lines.append(line)
                    # 提取标题
                    if stripped.startswith('#'):
                        title_match = re.match(r'^#+\s+(.+)$', stripped)
                        if title_match and not current_title:
                            current_title = title_match.group(1).strip()
            
            # 保存最后一个页面
            if current_page_lines:
                page = AIService._extract_page_from_lines(current_page_lines, current_title)
                if page:
                    pages.append(page)
        else:
            # 检查是否有标题层级
            has_h1 = any(line.strip().startswith('# ') for line in lines)
            has_h2 = any(line.strip().startswith('## ') for line in lines)
            
            if has_h1:
                # 使用 # 一级标题作为页面分隔
                logger.info("检测到一级标题 (#)，使用一级标题分页")
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('# '):
                        # 遇到一级标题，保存当前页面
                        if current_page_lines:
                            page = AIService._extract_page_from_lines(current_page_lines, current_title)
                            if page:
                                pages.append(page)
                            current_page_lines = []
                        # 提取新标题
                        current_title = stripped[2:].strip()
                    current_page_lines.append(line)
                
                # 保存最后一个页面
                if current_page_lines:
                    page = AIService._extract_page_from_lines(current_page_lines, current_title)
                    if page:
                        pages.append(page)
            elif has_h2:
                # 使用 ## 二级标题作为页面分隔
                logger.info("检测到二级标题 (##)，使用二级标题分页")
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('## '):
                        # 遇到二级标题，保存当前页面
                        if current_page_lines:
                            page = AIService._extract_page_from_lines(current_page_lines, current_title)
                            if page:
                                pages.append(page)
                            current_page_lines = []
                        # 提取新标题
                        current_title = stripped[3:].strip()
                    current_page_lines.append(line)
                
                # 保存最后一个页面
                if current_page_lines:
                    page = AIService._extract_page_from_lines(current_page_lines, current_title)
                    if page:
                        pages.append(page)
            else:
                # 没有明确的分隔符或标题，返回 None 让 AI 处理
                logger.info("未检测到明确的分隔符或标题层级")
                return None
        
        # 如果成功识别到至少一个页面，返回结果
        if pages:
            return pages
        
        return None
    
    @staticmethod
    def _extract_page_from_lines(lines: List[str], title: Optional[str]) -> Optional[Dict]:
        """
        从行列表中提取页面信息
        
        Args:
            lines: 页面内容的行列表
            title: 页面标题（可选）
        
        Returns:
            页面字典，包含 title, points, description
        """
        # 移除空行
        content_lines = [line for line in lines if line.strip()]
        
        if not content_lines:
            return None
        
        # 重新组合内容
        full_content = '\n'.join(lines).strip()
        
        # 提取标题（如果没有提供）
        if not title:
            for line in content_lines:
                stripped = line.strip()
                # 尝试从一级或二级标题提取
                if stripped.startswith('# '):
                    title = stripped[2:].strip()
                    break
                elif stripped.startswith('## '):
                    title = stripped[3:].strip()
                    break
        
        # 如果还是没有标题，使用第一行非空内容或默认标题
        if not title:
            title = content_lines[0].strip()[:50] if content_lines else "未命名页面"
        
        # 提取要点（bullet points）
        points = []
        for line in content_lines:
            stripped = line.strip()
            # 匹配各种列表格式
            if stripped.startswith('-') or stripped.startswith('*') or stripped.startswith('•'):
                point = stripped[1:].strip()
                if point:
                    points.append(point)
            elif re.match(r'^\d+[.)、]\s+', stripped):
                # 数字列表：1. 或 1) 或 1、
                point = re.sub(r'^\d+[.)、]\s+', '', stripped)
                if point:
                    points.append(point)
        
        # 构建描述内容（保持原格式）
        description = f"页面标题：{title}\n\n页面文字：\n{full_content}"
        
        return {
            'title': title,
            'points': points if points else ['原文内容'],
            'description': description
        }


