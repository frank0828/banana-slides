import React, { useState, useEffect, useRef } from 'react';
import { Image as ImageIcon, ImagePlus, Upload, X, FolderOpen, Sparkles, Wand2 } from 'lucide-react';
import { Modal, Textarea, Button, useToast, MaterialSelector, Skeleton } from '@/components/shared';
import { generateMaterialImage, generateAdMaterialImage, getTaskStatus } from '@/api/endpoints';
import { getImageUrl } from '@/api/client';
import { materialUrlToFile } from './MaterialSelector';
import type { Material } from '@/api/endpoints';
import type { Task } from '@/types';

interface MaterialGeneratorModalProps {
  projectId?: string | null; // 可选，如果不提供则生成全局素材
  isOpen: boolean;
  onClose: () => void;
}

type GeneratorMode = 'raw' | 'ecom';

/**
 * 素材生成模态卡片
 * - 通用模式：提示词原样传给文生图模型
 * - 电商广告模式：按结构化配置生成电商广告图
 *   - AI 自由发挥开关：开启后 AI 全权决定视觉方向
 *   - 风格参考图：独立于商品图，用于指定视觉风格
 *   - 自定义补充描述：追加自由文本到 prompt
 * - 生成结果保存到素材库
 */
export const MaterialGeneratorModal: React.FC<MaterialGeneratorModalProps> = ({
  projectId,
  isOpen,
  onClose,
}) => {
  const { show } = useToast();
  const [mode, setMode] = useState<GeneratorMode>('raw');

  // 通用模式
  const [prompt, setPrompt] = useState('');
  const [refImage, setRefImage] = useState<File | null>(null);
  const [extraImages, setExtraImages] = useState<File[]>([]);

  // 电商广告模式 ── 基础信息
  const [productName, setProductName] = useState('');
  const [sellingPoints, setSellingPoints] = useState('');
  const [price, setPrice] = useState('');
  const [targetAudience, setTargetAudience] = useState('');
  // 电商广告模式 ── 视觉配置
  const [visualStyle, setVisualStyle] = useState('');
  const [scene, setScene] = useState('');
  const [mood, setMood] = useState('');
  const [aspectRatio, setAspectRatio] = useState('1:1');
  const [resolution, setResolution] = useState('2K');
  const [composition, setComposition] = useState('center-product');
  const [textDensity, setTextDensity] = useState('medium');
  const [tone, setTone] = useState('');
  // 电商广告模式 ── 新增灵活参数
  const [aiCreativeMode, setAiCreativeMode] = useState(false);
  const [customPrompt, setCustomPrompt] = useState('');
  const [productImages, setProductImages] = useState<File[]>([]);
  const [styleRefImages, setStyleRefImages] = useState<File[]>([]);

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const elapsedTimerRef = useRef<NodeJS.Timeout | null>(null);
  const [isMaterialSelectorOpen, setIsMaterialSelectorOpen] = useState(false);

  const handleRefImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = (e.target.files && e.target.files[0]) || null;
    if (file) {
      setRefImage(file);
    }
  };

  const handleExtraImagesChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    if (!refImage) {
      const [first, ...rest] = files;
      setRefImage(first);
      if (rest.length > 0) {
        setExtraImages((prev) => [...prev, ...rest]);
      }
    } else {
      setExtraImages((prev) => [...prev, ...files]);
    }
  };

  const removeExtraImage = (index: number) => {
    setExtraImages((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSelectMaterials = async (materials: Material[]) => {
    try {
      const files = await Promise.all(
        materials.map((material) => materialUrlToFile(material))
      );

      if (files.length === 0) return;

      if (!refImage) {
        const [first, ...rest] = files;
        setRefImage(first);
        if (rest.length > 0) {
          setExtraImages((prev) => [...prev, ...rest]);
        }
      } else {
        setExtraImages((prev) => [...prev, ...files]);
      }

      show({ message: `已添加 ${files.length} 个素材`, type: 'success' });
    } catch (error: any) {
      console.error('加载素材失败:', error);
      show({
        message: '加载素材失败: ' + (error.message || '未知错误'),
        type: 'error',
      });
    }
  };

  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
      if (elapsedTimerRef.current) {
        clearInterval(elapsedTimerRef.current);
      }
    };
  }, []);

  // 每次打开弹窗时清空上次的预览结果
  useEffect(() => {
    if (isOpen) {
      setPreviewUrl(null);
      setIsGenerating(false);
    }
  }, [isOpen]);

  const pollMaterialTask = async (taskId: string) => {
    const targetProjectId = projectId || 'global';
    const maxAttempts = 90;  // 每2秒一次，最多等 3 分钟
    let attempts = 0;

    // 启动计时器
    setElapsedSeconds(0);
    if (elapsedTimerRef.current) clearInterval(elapsedTimerRef.current);
    elapsedTimerRef.current = setInterval(() => {
      setElapsedSeconds(prev => prev + 1);
    }, 1000);

    const stopPolling = () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
      if (elapsedTimerRef.current) {
        clearInterval(elapsedTimerRef.current);
        elapsedTimerRef.current = null;
      }
    };

    const poll = async () => {
      try {
        attempts++;
        const response = await getTaskStatus(targetProjectId, taskId);
        const task = response.data as Task;

        if (task.status === 'COMPLETED') {
          const progress = (task.progress || {}) as Record<string, string>;
          const imageUrl = progress.image_url;

          if (imageUrl) {
            setPreviewUrl(getImageUrl(imageUrl));
            const message = projectId
              ? '素材生成成功，已保存到历史素材库'
              : '素材生成成功，已保存到全局素材库';
            show({ message, type: 'success' });
          } else {
            show({ message: '素材生成完成，但未找到图片地址', type: 'error' });
          }

          setIsGenerating(false);
          stopPolling();
        } else if (task.status === 'FAILED') {
          show({
            message: task.error_message || '素材生成失败',
            type: 'error',
          });
          setIsGenerating(false);
          stopPolling();
        } else if (task.status === 'PENDING' || task.status === 'RUNNING' || task.status === 'PROCESSING') {
          if (attempts >= maxAttempts) {
            show({ message: '素材生成超时（超过3分钟），请稍后查看素材库', type: 'info' });
            setIsGenerating(false);
            stopPolling();
          }
        }
      } catch (error: any) {
        console.error('轮询任务状态失败:', error);
        if (attempts >= maxAttempts) {
          show({ message: '轮询任务状态失败，请稍后查看素材库', type: 'error' });
          setIsGenerating(false);
          stopPolling();
        }
      }
    };

    poll();
    pollingIntervalRef.current = setInterval(poll, 2000);
  };

  const handleGenerate = async () => {
    if (mode === 'raw') {
      // 通用模式：保持原有逻辑
      if (!prompt.trim()) {
        show({ message: '请输入提示词', type: 'error' });
        return;
      }

      setIsGenerating(true);
      setPreviewUrl(null);
      try {
        const targetProjectId = projectId || 'none';
        const resp = await generateMaterialImage(
          targetProjectId,
          prompt.trim(),
          refImage as File,
          extraImages
        );
        const taskId = resp.data?.task_id;

        if (taskId) {
          await pollMaterialTask(taskId);
        } else {
          show({ message: '素材生成失败：未返回任务ID', type: 'error' });
          setIsGenerating(false);
        }
      } catch (error: any) {
        show({
          message: error?.response?.data?.error?.message || error.message || '素材生成失败',
          type: 'error',
        });
        setIsGenerating(false);
      }
      return;
    }

    // 电商广告模式
    if (!productName.trim()) {
      show({ message: '请填写商品名称', type: 'error' });
      return;
    }

    setIsGenerating(true);
    setPreviewUrl(null);
    try {
      const targetProjectId = projectId || 'none';
      const sellingPointsList = sellingPoints
        .split('\n')
        .map((s) => s.trim())
        .filter(Boolean);

      const config = {
        product: {
          name: productName.trim(),
          selling_points: sellingPointsList,
          price: price.trim() || undefined,
          target_audience: targetAudience.trim() || undefined,
        },
        // AI 自由发挥模式下，style/layout/tone 仍可传递，但 prompt 会忽略
        style: aiCreativeMode ? undefined : {
          visual_style: visualStyle.trim() || '电商风格，清晰突出商品',
          scene: scene.trim() || undefined,
          mood: mood.trim() || undefined,
        },
        layout: aiCreativeMode ? { aspect_ratio: aspectRatio, resolution } : {
          aspect_ratio: aspectRatio,
          resolution,
          composition,
          text_density: textDensity,
        },
        tone: aiCreativeMode ? undefined : (tone.trim() || '简洁有力的电商文案风格'),
        ai_creative_mode: aiCreativeMode,
        custom_prompt: customPrompt.trim() || undefined,
      };

      const resp = await generateAdMaterialImage(
        targetProjectId,
        config,
        productImages.length > 0 ? productImages : undefined,
        styleRefImages.length > 0 ? styleRefImages : undefined,
      );
      const taskId = resp.data?.task_id;

      if (taskId) {
        await pollMaterialTask(taskId);
      } else {
        show({ message: '素材生成失败：未返回任务ID', type: 'error' });
        setIsGenerating(false);
      }
    } catch (error: any) {
      show({
        message: error?.response?.data?.error?.message || error.message || '素材生成失败',
        type: 'error',
      });
      setIsGenerating(false);
    }
  };

  const handleClose = () => {
    onClose();
  };

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="素材生成" size="lg">
      <blockquote className="text-sm text-gray-500 mb-4">生成的素材会保存到素材库</blockquote>
      <div className="space-y-4">
        {/* 顶部：生成结果预览 */}
        <div className="bg-gray-50 rounded-lg border border-gray-200 p-4">
          <h4 className="text-sm font-semibold text-gray-700 mb-2">生成结果</h4>
          {isGenerating ? (
            <div className="aspect-video rounded-lg overflow-hidden border border-gray-200 relative">
              <Skeleton className="w-full h-full" />
              <div className="absolute bottom-2 left-0 right-0 flex justify-center">
                <span className="bg-black/50 text-white text-xs px-2 py-1 rounded-full">
                  AI 生成中... 已等待 {elapsedSeconds}s
                </span>
              </div>
            </div>
          ) : previewUrl ? (
            <div className="aspect-video bg-white rounded-lg overflow-hidden border border-gray-200 flex items-center justify-center">
              <img
                src={previewUrl}
                alt="生成的素材"
                className="w-full h-full object-contain"
              />
            </div>
          ) : (
            <div className="aspect-video bg-gray-100 rounded-lg flex flex-col items-center justify-center text-gray-400 text-sm">
              <div className="text-3xl mb-2">🎨</div>
              <div>生成的素材会展示在这里</div>
            </div>
          )}
        </div>

        {/* 模式切换 */}
        <div className="inline-flex items-center rounded-full bg-gray-100 p-1 text-xs">
          <button
            type="button"
            className={`px-3 py-1 rounded-full transition-colors ${
              mode === 'raw' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-500'
            }`}
            onClick={() => setMode('raw')}
          >
            通用模式
          </button>
          <button
            type="button"
            className={`px-3 py-1 rounded-full transition-colors ${
              mode === 'ecom' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-500'
            }`}
            onClick={() => setMode('ecom')}
          >
            电商广告模式
          </button>
        </div>

        {/* 通用模式：提示词输入 */}
        {mode === 'raw' ? (
          <Textarea
            label="提示词（原样发送给文生图模型）"
            placeholder="例如：蓝紫色渐变背景，带几何图形和科技感线条，用于科技主题标题页..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
          />
        ) : (
          /* 电商广告模式：结构化配置 */
          <div className="space-y-4">
            {/* ── AI 自由发挥开关 ── */}
            <div
              className={`flex items-center justify-between rounded-lg border px-3 py-2.5 cursor-pointer transition-colors select-none ${
                aiCreativeMode
                  ? 'border-banana-400 bg-banana-50'
                  : 'border-gray-200 bg-gray-50 hover:border-gray-300'
              }`}
              onClick={() => setAiCreativeMode((v) => !v)}
            >
              <div className="flex items-center gap-2">
                <Wand2 size={15} className={aiCreativeMode ? 'text-banana-600' : 'text-gray-400'} />
                <div>
                  <div className={`text-xs font-semibold ${aiCreativeMode ? 'text-banana-700' : 'text-gray-700'}`}>
                    AI 自由发挥模式
                  </div>
                  <div className="text-[11px] text-gray-400 mt-0.5">
                    {aiCreativeMode
                      ? '已开启：AI 将自主决定视觉风格、构图、配色等所有出图方向'
                      : '开启后 AI 自行决定所有视觉参数，适合不确定风格时探索'}
                  </div>
                </div>
              </div>
              {/* 开关 */}
              <div
                className={`w-9 h-5 rounded-full transition-colors flex-shrink-0 flex items-center px-0.5 ${
                  aiCreativeMode ? 'bg-banana-500' : 'bg-gray-300'
                }`}
              >
                <div
                  className={`w-4 h-4 bg-white rounded-full shadow transition-transform ${
                    aiCreativeMode ? 'translate-x-4' : 'translate-x-0'
                  }`}
                />
              </div>
            </div>

            {/* ── 基础信息（始终显示） ── */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  商品名称 <span className="text-red-500">*</span>
                </label>
                <input
                  className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-banana-500 focus:border-banana-500"
                  placeholder="例如：蓝牙降噪耳机"
                  value={productName}
                  onChange={(e) => setProductName(e.target.value)}
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  价格 / 标签（可选）
                </label>
                <input
                  className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-banana-500 focus:border-banana-500"
                  placeholder="例如：¥199 限时直降"
                  value={price}
                  onChange={(e) => setPrice(e.target.value)}
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                核心卖点（每行一条，可选）
              </label>
              <textarea
                className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-banana-500 focus:border-banana-500 resize-none"
                placeholder={"例如：\n48dB 深度降噪\n30 小时续航\n轻量化折叠设计"}
                rows={3}
                value={sellingPoints}
                onChange={(e) => setSellingPoints(e.target.value)}
              />
            </div>

            {/* ── 图片尺寸（始终显示） ── */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">图片比例</label>
                <select
                  className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-banana-500 focus:border-banana-500"
                  value={aspectRatio}
                  onChange={(e) => setAspectRatio(e.target.value)}
                >
                  <option value="1:1">1:1（方图）</option>
                  <option value="3:4">3:4（竖版）</option>
                  <option value="4:3">4:3（横版）</option>
                  <option value="9:16">9:16（竖版封面）</option>
                  <option value="16:9">16:9（横版海报）</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">图片尺寸</label>
                <select
                  className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-banana-500 focus:border-banana-500"
                  value={resolution}
                  onChange={(e) => setResolution(e.target.value)}
                >
                  <option value="1K">1K（快速，约 1024px）</option>
                  <option value="2K">2K（推荐，约 2048px）</option>
                  <option value="4K">4K（高清，约 4096px）</option>
                </select>
              </div>
            </div>

            {/* ── 结构化视觉参数（AI 自由发挥时折叠/隐藏） ── */}
            {!aiCreativeMode && (
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">
                      目标人群（可选）
                    </label>
                    <input
                      className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-banana-500 focus:border-banana-500"
                      placeholder="例如：通勤上班族、学生党"
                      value={targetAudience}
                      onChange={(e) => setTargetAudience(e.target.value)}
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">
                      画面风格（可选）
                    </label>
                    <input
                      className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-banana-500 focus:border-banana-500"
                      placeholder="例如：科技感、蓝黑配色、强对比"
                      value={visualStyle}
                      onChange={(e) => setVisualStyle(e.target.value)}
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">
                      场景（可选）
                    </label>
                    <input
                      className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-banana-500 focus:border-banana-500"
                      placeholder="例如：地铁通勤、办公桌面"
                      value={scene}
                      onChange={(e) => setScene(e.target.value)}
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">
                      氛围（可选）
                    </label>
                    <input
                      className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-banana-500 focus:border-banana-500"
                      placeholder="例如：安静、专注、轻松"
                      value={mood}
                      onChange={(e) => setMood(e.target.value)}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">构图</label>
                    <select
                      className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-banana-500 focus:border-banana-500"
                      value={composition}
                      onChange={(e) => setComposition(e.target.value)}
                    >
                      <option value="center-product">商品居中</option>
                      <option value="left-image-right-text">左图右文</option>
                      <option value="right-image-left-text">右图左文</option>
                      <option value="top-image-bottom-text">上图下文</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">文本占比</label>
                    <select
                      className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-banana-500 focus:border-banana-500"
                      value={textDensity}
                      onChange={(e) => setTextDensity(e.target.value)}
                    >
                      <option value="low">偏少</option>
                      <option value="medium">适中</option>
                      <option value="high">偏多</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    文案语气（可选）
                  </label>
                  <input
                    className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-banana-500 focus:border-banana-500"
                    placeholder="例如：种草风、官方口吻、硬核参数党"
                    value={tone}
                    onChange={(e) => setTone(e.target.value)}
                  />
                </div>
              </>
            )}

            {/* ── 自定义补充描述（始终显示） ── */}
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                <span className="flex items-center gap-1">
                  <Sparkles size={12} className="text-banana-500" />
                  自定义补充描述（可选）
                </span>
              </label>
              <textarea
                className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-banana-500 focus:border-banana-500 resize-none"
                placeholder="例如：整体色调偏暗金色，要有高端奢华感；右下角加二维码占位；文字用白色..."
                rows={2}
                value={customPrompt}
                onChange={(e) => setCustomPrompt(e.target.value)}
              />
            </div>

            {/* ── 商品图片上传 ── */}
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-2">
                商品图片（可选，上传后 AI 将以此为参考生成）
              </label>
              <div className="flex flex-wrap gap-2">
                {productImages.map((file, idx) => (
                  <div key={idx} className="relative w-20 h-20 rounded border border-gray-200 overflow-hidden group">
                    <img
                      src={URL.createObjectURL(file)}
                      alt={`商品图${idx + 1}`}
                      className="w-full h-full object-cover"
                    />
                    <button
                      type="button"
                      className="absolute top-0.5 right-0.5 bg-black/50 text-white rounded-full w-4 h-4 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                      onClick={() => setProductImages((prev) => prev.filter((_, i) => i !== idx))}
                    >
                      <X size={10} />
                    </button>
                  </div>
                ))}
                {productImages.length < 4 && (
                  <label className="w-20 h-20 border-2 border-dashed border-gray-300 rounded flex flex-col items-center justify-center cursor-pointer hover:border-banana-500 transition-colors bg-gray-50 text-gray-400">
                    <Upload size={16} className="mb-1" />
                    <span className="text-xs">上传图片</span>
                    <input
                      type="file"
                      accept="image/*"
                      multiple
                      className="hidden"
                      onChange={(e) => {
                        const files = Array.from(e.target.files || []);
                        setProductImages((prev) => [...prev, ...files].slice(0, 4));
                        e.target.value = '';
                      }}
                    />
                  </label>
                )}
              </div>
              <p className="text-xs text-gray-400 mt-1">最多上传 4 张，建议使用纯色背景的商品白底图</p>
            </div>

            {/* ── 风格参考图上传（独立于商品图） ── */}
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-2">
                <span className="flex items-center gap-1">
                  <ImagePlus size={12} className="text-gray-500" />
                  风格参考图（可选，AI 会模仿其视觉风格 / 构图）
                </span>
              </label>
              <div className="flex flex-wrap gap-2">
                {styleRefImages.map((file, idx) => (
                  <div key={idx} className="relative w-20 h-20 rounded border border-purple-200 overflow-hidden group">
                    <img
                      src={URL.createObjectURL(file)}
                      alt={`风格参考图${idx + 1}`}
                      className="w-full h-full object-cover"
                    />
                    {/* 紫色标识角标 */}
                    <div className="absolute bottom-0 left-0 right-0 bg-purple-500/60 text-white text-[9px] text-center py-0.5">
                      风格参考
                    </div>
                    <button
                      type="button"
                      className="absolute top-0.5 right-0.5 bg-black/50 text-white rounded-full w-4 h-4 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                      onClick={() => setStyleRefImages((prev) => prev.filter((_, i) => i !== idx))}
                    >
                      <X size={10} />
                    </button>
                  </div>
                ))}
                {styleRefImages.length < 4 && (
                  <label className="w-20 h-20 border-2 border-dashed border-purple-300 rounded flex flex-col items-center justify-center cursor-pointer hover:border-purple-500 transition-colors bg-purple-50 text-purple-400">
                    <ImagePlus size={16} className="mb-1" />
                    <span className="text-xs">上传</span>
                    <input
                      type="file"
                      accept="image/*"
                      multiple
                      className="hidden"
                      onChange={(e) => {
                        const files = Array.from(e.target.files || []);
                        setStyleRefImages((prev) => [...prev, ...files].slice(0, 4));
                        e.target.value = '';
                      }}
                    />
                  </label>
                )}
              </div>
              <p className="text-xs text-gray-400 mt-1">
                最多上传 4 张，用于指定整体视觉风格/色调，与商品图独立处理
              </p>
            </div>
          </div>
        )}

        {/* 参考图上传区（通用模式下显示） */}
        {mode === 'raw' && (
          <div className="bg-gray-50 rounded-lg border border-gray-200 p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm text-gray-700">
                <ImagePlus size={16} className="text-gray-500" />
                <span className="font-medium">参考图片（可选）</span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                icon={<FolderOpen size={16} />}
                onClick={() => setIsMaterialSelectorOpen(true)}
              >
                从素材库选择
              </Button>
            </div>
            <div className="flex flex-wrap gap-4">
              {/* 主参考图 */}
              <div className="space-y-2">
                <div className="text-xs text-gray-600">主参考图（可选）</div>
                <label className="w-40 h-28 border-2 border-dashed border-gray-300 rounded flex flex-col items-center justify-center cursor-pointer hover:border-banana-500 transition-colors bg-white relative group">
                  {refImage ? (
                    <>
                      <img
                        src={URL.createObjectURL(refImage)}
                        alt="主参考图"
                        className="w-full h-full object-cover"
                      />
                      <button
                        type="button"
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          setRefImage(null);
                        }}
                        className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 text-white rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity shadow z-10"
                      >
                        <X size={12} />
                      </button>
                    </>
                  ) : (
                    <>
                      <ImageIcon size={24} className="text-gray-400 mb-1" />
                      <span className="text-xs text-gray-500">点击上传</span>
                    </>
                  )}
                  <input
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleRefImageChange}
                  />
                </label>
              </div>

              {/* 额外参考图 */}
              <div className="flex-1 space-y-2 min-w-[180px]">
                <div className="text-xs text-gray-600">额外参考图（可选，多张）</div>
                <div className="flex flex-wrap gap-2">
                  {extraImages.map((file, idx) => (
                    <div key={idx} className="relative group">
                      <img
                        src={URL.createObjectURL(file)}
                        alt={`extra-${idx + 1}`}
                        className="w-20 h-20 object-cover rounded border border-gray-300"
                      />
                      <button
                        onClick={() => removeExtraImage(idx)}
                        className="absolute -top-2 -right-2 w-5 h-5 bg-red-500 text-white rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <X size={12} />
                      </button>
                    </div>
                  ))}
                  <label className="w-20 h-20 border-2 border-dashed border-gray-300 rounded flex flex-col items-center justify-center cursor-pointer hover:border-banana-500 transition-colors bg-white">
                    <Upload size={18} className="text-gray-400 mb-1" />
                    <span className="text-[11px] text-gray-500">添加</span>
                    <input
                      type="file"
                      accept="image/*"
                      multiple
                      className="hidden"
                      onChange={handleExtraImagesChange}
                    />
                  </label>
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="flex justify-end gap-3 pt-2">
          <Button variant="ghost" onClick={handleClose} disabled={isGenerating}>
            关闭
          </Button>
          <Button
            variant="primary"
            onClick={handleGenerate}
            disabled={
              isGenerating ||
              (mode === 'raw' ? !prompt.trim() : !productName.trim())
            }
          >
            {isGenerating
              ? '生成中...'
              : mode === 'raw'
              ? '生成素材'
              : aiCreativeMode
              ? '✨ AI 自由发挥生成'
              : '生成电商广告图'}
          </Button>
        </div>
      </div>

      {/* 素材选择器（通用模式专用） */}
      <MaterialSelector
        projectId={projectId ?? undefined}
        isOpen={isMaterialSelectorOpen}
        onClose={() => setIsMaterialSelectorOpen(false)}
        onSelect={handleSelectMaterials}
        multiple={true}
      />
    </Modal>
  );
};
