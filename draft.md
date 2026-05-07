3 天的话，目标要改成“高质量 prototype + convincing evaluation”，不要做真正大规模训练。A100/H100 可以用来快速跑强 baseline、生成样本和做少量 LoRA/adapter 实验，但主线必须保证能交付。

**3 天最稳方案**
题目保留：`Layered Typography-Aware Poster Generation with Editable Text`

核心 claim：

> Diffusion models still struggle with exact text and editable typography. We use a layered generation pipeline: model generates visual background, structured layout module places text, deterministic renderer guarantees spelling, and optional image harmonization improves integration.

这不是“我们训练出一个新大模型”，而是“我们提出一个结构化 pipeline，解决 text-heavy visual 的可控性和可编辑性”。

**Day 1：跑 baseline + 搭 pipeline**
上午：

- 准备 20-30 个 prompt：
  - movie poster
  - travel poster
  - coffee shop ad
  - concert flyer
  - book cover
  - museum exhibition poster
  - product sale banner
- 每个 prompt 指定 2-4 行文字。
- 用 Qwen-Image / Qwen-Image-Edit、AnyText2、SDXL 或 FLUX 生成 baseline。
- 保存失败案例：错字、乱码、文字缺失、排版差。

下午：

- 写 deterministic poster renderer：
  - 背景图来自 T2I 模型。
  - 文字层用 PIL / OpenCV 渲染。
  - 自动计算 title/subtitle/date 的 bbox。
  - 自动选颜色：从背景取 palette，保证 contrast。
  - 加 stroke/shadow/transparent overlay。
- 输出 layout JSON。

晚上：

- 做第一批效果图：
  - baseline direct generation
  - ours layered rendering
  - bbox overlay
  - edit example

这一天结束必须已经有“看起来能赢”的图。

**Day 2：做 saliency-aware layout + metrics**
上午：

- 用现成 saliency / segmentation 模型避免文字压主体：
  - 简单版：用 CLIPSeg / SAM / rembg / edge density。
  - 更简单版：把画面划成候选区域，选低纹理、低 saliency、高 contrast 的区域放字。
- 实现 layout scoring：
  - 不出界
  - bbox 不重叠
  - 文字区域背景不能太复杂
  - title 最大、subtitle 次之、metadata 最小

下午：

- 做 evaluation：
  - OCR accuracy：baseline vs ours。
  - text exact match rate。
  - edit background preservation：非文字区域 SSIM / LPIPS。
  - layout validity：overlap rate / out-of-bound rate。
- 跑 20-30 张足够。

晚上：

- 做 ablation 图：
  - baseline diffusion
  - renderer without saliency
  - renderer without contrast
  - full method
- 如果还有时间，用 A100/H100 跑一个轻量 experiment：
  - 不建议训练大模型。
  - 可以训练/微调一个 layout ranker 或 LoRA 小实验作为 bonus。
  - 但不要让它成为主线依赖。

**Day 3：整理 final artifacts**
上午：

- 精挑 6-10 张最好看的 qualitative results。
- 做一张 pipeline figure。
- 做一张 quantitative table。
- 做一张 editability figure。

下午：

- 写 report：
  - problem：T2I text is not reliable/editable
  - gap：AnyText/Qwen improves rendering but still raster-entangled or less structured
  - method：layered pipeline
  - results：OCR/edit/layout/aesthetic examples
  - limitations：not end-to-end, depends on renderer/fonts, limited free-form curved text

晚上：

- 最后跑一次全部脚本。
- 固定 random seed。
- 保存 outputs、figures、JSON、PDF。

**算力怎么用**
A100/H100 用法优先级：

1. 跑 Qwen-Image / Qwen-Image-Edit baseline 和 strong examples。
2. 批量生成 100-200 张背景图。
3. 跑 OCR、CLIP、saliency、LPIPS。
4. 可选：训练小 layout ranker。

不要在 3 天内尝试：

- 从头训练 diffusion。
- 大规模 fine-tune Qwen/FLUX。
- 自己训 OCR。
- 做复杂 UI。

**最小可交付结果**
必须有：

- `generate_backgrounds.py`
- `render_poster.py`
- `layout.py`
- `evaluate.py`
- `outputs/baselines/`
- `outputs/ours/`
- `figures/pipeline.png`
- `figures/qualitative_grid.png`
- `figures/edit_examples.png`
- `results/metrics.csv`

**论文里最强的结果表可以长这样**

| Method | OCR Exact Match ↑ | Layout Valid ↑ | Edit BG SSIM ↑ | Human Preference ↑ |
|---|---:|---:|---:|---:|
| SDXL direct | low | medium | N/A | low |
| Qwen/AnyText direct | medium-high | medium | low/medium | medium |
| Ours layered | near 100% | high | high | high |

你的优势是：文字 exact match 几乎必胜，因为文字是 renderer 保证的。project 的重点就变成“如何让 renderer 的文字看起来不是简单贴图，并且 layout 合理”。

**一句话策略**
3 天内别赌模型会学会写字；让模型负责视觉，让结构化 pipeline 负责文字。这样最终效果图一定能好看，评估也能站得住。