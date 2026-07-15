# Zotero Ultimate OCR Knowledge Base

把 Zotero 导出的 PDF 与 BibTeX 转换成可供人阅读、可供 AI/RAG 检索、并且能反查原始引用的本地知识库。

项目提供两个脚本：

- `scripts/ocr_zotero.py`：递归发现 Zotero PDF，使用百度 Unlimited-OCR API 或本地 `baidu/Unlimited-OCR` 模型识别，并生成统一格式的 OCR 结果。
- `scripts/build_knowledge_base.py`：关联 OCR、BibTeX 与 PDF 附件，生成 Markdown 文档、总索引、清单及 RAG 分块。

两种 OCR 后端的落盘格式完全相同，因此可以随时切换，也可以让部分文献走 API、部分文献走本地模型。

## 工作流程

```text
Zotero export (.bib + files/**/*.pdf)
                    |
                    v
      ocr_zotero.py baidu | local
                    |
                    v
        ocr_results/*/{meta.json,result.md}
                    |
                    v
       build_knowledge_base.py
                    |
                    v
 knowledge_base/{index.md,manifest.jsonl,chunks.jsonl,docs/}
```

## 功能

- 递归扫描一个或多个 Zotero 导出目录，只处理 PDF。
- 百度 API：提交、保存任务 ID、轮询、失败续跑、立即下载 Markdown/JSON。
- 本地模型：从 Hugging Face、本机缓存或指定目录加载 Unlimited-OCR。
- 已完成文件默认跳过；文档目录名包含路径哈希，不会因同名 PDF 相互覆盖。
- 通过附件绝对路径优先匹配 OCR，必要时再以唯一文件名匹配。
- 每篇文献保留 citekey、作者、年份、DOI、URL、摘要、附件路径及原始 BibTeX。
- 输出 `chunks.jsonl`，可直接作为向量化或 RAG 入库前的分块数据。
- 没有 OCR 的条目仍会生成元数据文档，并明确标记 `ocr_status=missing`。
- 未匹配回 BibTeX 的 OCR 结果写入 `unmatched_ocr.json`，方便审计。

## 环境要求

- Python 3.10+
- 百度后端只使用 Python 标准库，不需要安装第三方包。
- 本地后端需要 PyTorch、Transformers、PyMuPDF 及模型依赖；官方示例以 NVIDIA CUDA GPU 运行。CPU 模式可以启动，但处理 3B 模型通常很慢。

建议使用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

只使用百度 API 时无需继续安装依赖。本地运行时安装：

```bash
pip install -r requirements-local.txt
```

`requirements-local.txt` 按 [Unlimited-OCR 官方模型卡](https://huggingface.co/baidu/Unlimited-OCR)列出的已测试版本固定。不同 CUDA 环境可能需要先按 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)安装匹配的 `torch` / `torchvision`，再安装其余依赖。

## 准备 Zotero 导出

在 Zotero 中选择文献或分类，执行“导出文献库”：

1. 格式选择 `BibTeX`。
2. 勾选“导出文件”。
3. 将导出结果放在同一目录中；`.bib` 的 `file` 字段应指向 `files/.../*.pdf`。

典型目录如下：

```text
library/
├── library.bib
└── files/
    ├── 1234/paper-a.pdf
    └── 5678/paper-b.pdf
```

以下示例都假定在本仓库根目录执行命令。脚本也接受单个 PDF、单个 `.bib` 或多个导出目录。

## 方式一：百度 Unlimited-OCR API

百度接口需要 OCR 应用的 API Key 与 Secret Key。密钥只通过环境变量读取，不要写入代码、README 或提交到 Git：

```bash
export BAIDU_OCR_API_KEY='your-api-key'
export BAIDU_OCR_SECRET_KEY='your-secret-key'
```

先检查将要处理的 PDF：

```bash
python scripts/ocr_zotero.py baidu library --dry-run
```

处理全部文件：

```bash
python scripts/ocr_zotero.py baidu library --output-dir ocr_results
```

脚本会逐篇提交、轮询并下载结果。根据[百度 Unlimited-OCR API 文档](https://ai.baidu.com/ai-doc/OCR/fmr1p39gb)，该接口为异步接口，任务成功后返回的 Markdown/JSON URL 有效期有限；本脚本会在成功后立即保存到本地。

### 分批提交与续跑

先测试两篇：

```bash
python scripts/ocr_zotero.py baidu library --limit 2
```

只提交、不等待结果：

```bash
python scripts/ocr_zotero.py baidu library --submit-only
```

稍后再次运行同一命令（去掉 `--submit-only`）。脚本会从 `meta.json` 读取已有 `task_id` 并继续轮询，不会重复提交。已经存在 `result.md` 的文档会自动跳过；只有明确传入 `--force` 才会重新提交。

轮询时间也可以调整：

```bash
python scripts/ocr_zotero.py baidu library \
  --poll-interval 10 \
  --timeout 3600
```

### 大文件和 `file_url`

默认 `--file-mode data` 会把 PDF 以 base64 表单提交。脚本将 50 MiB 作为 `file_data` 安全上限；更大的文件请使用可公开下载的 URL：

```bash
python scripts/ocr_zotero.py baidu library \
  --file-mode auto \
  --url-base 'https://example.com/zotero-export/'
```

`--url-base` 要求远端路径与当前工作目录内的相对路径一致。若 URL 不规则，提供 JSON 映射：

```json
{
  "library/files/1234/paper-a.pdf": "https://example.com/download/a.pdf"
}
```

```bash
python scripts/ocr_zotero.py baidu library \
  --file-mode url \
  --url-map url-map.json
```

远端地址必须能被百度服务器直接访问，并关闭防盗链。不要把带长期访问凭据的 URL 映射提交到公开仓库。

## 方式二：本地 Unlimited-OCR

默认模型为 `baidu/Unlimited-OCR`。首次运行会从 Hugging Face 下载模型，之后复用缓存：

```bash
python scripts/ocr_zotero.py local library --output-dir ocr_results
```

显式选择 CUDA 和精度：

```bash
python scripts/ocr_zotero.py local library \
  --device cuda \
  --dtype bfloat16
```

使用已经下载好的模型目录并禁止联网：

```bash
python scripts/ocr_zotero.py local library \
  --model /data/models/Unlimited-OCR \
  --offline
```

也可以通过环境变量指定模型：

```bash
export UNLIMITED_OCR_MODEL='/data/models/Unlimited-OCR'
python scripts/ocr_zotero.py local library --offline
```

PDF 会先通过 PyMuPDF 渲染成图片，再调用模型的 `infer_multi()`。默认 200 DPI；提高 DPI 会显著增加显存、内存和耗时：

```bash
python scripts/ocr_zotero.py local library --dpi 300
```

页面图片默认使用临时目录并在每篇识别后删除。排查版面问题时可以保留：

```bash
python scripts/ocr_zotero.py local library --keep-pages
```

## 构建知识库

OCR 完成后运行：

```bash
python scripts/build_knowledge_base.py library \
  --ocr-dir ocr_results \
  --output-dir knowledge_base
```

多个 Zotero 导出库和多个 OCR 目录可以一起处理：

```bash
python scripts/build_knowledge_base.py kws references \
  --ocr-dir ocr_results-api \
  --ocr-dir ocr_results-local \
  --output-dir knowledge_base
```

即使尚未运行 OCR，也可以先建立只有元数据和摘要的知识库：

```bash
python scripts/build_knowledge_base.py library --output-dir knowledge_base
```

若只想收录已有 OCR 的文献：

```bash
python scripts/build_knowledge_base.py library --no-include-missing
```

调整 RAG 分块字符数与重叠：

```bash
python scripts/build_knowledge_base.py library \
  --chunk-size 3000 \
  --chunk-overlap 250
```

脚本每次重写本次涉及的文档、索引与 JSONL 清单，但不会删除输出目录中不再属于当前输入的旧 Markdown。更换数据集时建议使用新的 `--output-dir`，或先自行归档旧目录。

## 输出结构

### OCR 中间结果

```text
ocr_results/
└── paper-title-a1b2c3d4e5/
    ├── meta.json       # 原 PDF、后端、状态、task_id/模型信息
    ├── result.md       # 统一 OCR 正文
    ├── result.json     # 百度解析 JSON（若成功下载）
    └── pages/          # 仅本地模式加 --keep-pages 时存在
```

`meta.json` 是 OCR 与 Zotero 附件之间的主关联依据。请勿只复制 `result.md` 而丢失元数据。

### 知识库

```text
knowledge_base/
├── index.md                 # 人类/AI 可读总索引
├── manifest.jsonl           # 一行一篇文献的机器清单
├── chunks.jsonl             # 一行一个 RAG 文本块
├── unmatched_ocr.json       # 无法匹配到 BibTeX 的 OCR 输出
└── docs/
    └── <collection>/
        └── <citekey>.md      # 元数据、OCR 正文、原始 BibTeX
```

`chunks.jsonl` 的每行包含 `id`、`collection`、`citekey`、`title`、`year`、`doi`、`doc_path`、`chunk_index` 和 `text`。它不包含向量；可将其交给任意 embedding/向量数据库工具。

## 给 AI/RAG 的引用规则

建议把以下约束加入系统提示：

```text
只能使用 knowledge_base/docs、manifest.jsonl 和 chunks.jsonl 中的资料。
引用前必须核对 citekey、DOI、bib_source 和原始 BibTeX，不得根据标题猜测引用键。
OCR 可能存在识别错误；引用元数据以原始 BibTeX 为准。
ocr_status=missing 时，不得声称阅读过论文全文。
结论应附对应 citekey，并允许人工沿 PDF 附件路径复核。
```

## 常见问题

### 百度任务超时或网络中断

直接重跑原命令。已有 `task_id` 会被复用。若任务确实失败，错误会记录在 `meta.json`；确认原因后使用 `--force` 创建新任务。

### 百度成功但无法下载 Markdown

结果 URL 有有效期。若任务刚成功，重跑通常会重新查询并下载；如果 URL 已失效，可能需要 `--force` 重新提交。公开知识库中应提交已经落盘的结果，而不是依赖远端 URL。

### 本地模式 CUDA OOM

先降低 `--dpi`；关闭 `--keep-pages` 只能减少磁盘占用，不会降低推理显存。也可一次只传单个 PDF，或使用 `--limit` 分批处理。

### 知识库里 OCR 状态为 `missing`

检查：

- 对应 OCR 目录是否有 `meta.json` 和非空的 `result.md`；
- 构建时是否传入了正确的 `--ocr-dir`；
- Zotero `.bib` 的 `file` 字段是否仍指向导出后的 PDF；
- `knowledge_base/unmatched_ocr.json` 是否列出了该结果。

### 重复文件名是否会串文献

OCR 输出目录使用“文件名 + 原路径哈希”。建库时优先按绝对路径匹配；只有文件名在所有 OCR 结果中唯一时才回退到文件名匹配。无法唯一判断时保持 `missing`，不会静默选择其中一个。

### BibTeX 字段显示 LaTeX 花括号

脚本保留 BibTeX 字段原貌，以便引用审计，不负责完整的 LaTeX/Unicode 排版转换。这不影响 citekey、DOI、附件和 OCR 的关联。

## 安全与隐私

- 百度后端会将论文内容发送给第三方云服务；处理受限、保密或含个人信息的文档前，请确认授权和适用政策。
- 本地后端不会把 PDF 发送给百度 API，但首次从 Hugging Face 下载模型时仍会联网；使用 `--offline` 可强制只读本地模型/缓存。
- `trust_remote_code=True` 是该 Hugging Face 模型的加载要求。只应加载可信模型目录，并在高安全环境中审查或固定模型快照。
- 永远不要提交 API Key、Secret Key、访问令牌、私有下载 URL 或包含敏感全文的输出。
