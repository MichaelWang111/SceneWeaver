# SceneWeaver

SceneWeaver 是一个端到端的 AI 创意工作流项目，用来把真实商业片、招聘片、品牌短片拆解成可复用的导演经验，并把这些经验接入检索、脚本生成、意图推断、微调建议和素材生成流程。

它不直接生成视频。它做的是另一件更基础的事：把真实视频案例转化成结构化知识，让后续创意生成有证据、有来源、有可检索的经验参照。

```text
真实视频输入
-> 镜头拆分与抽帧
-> Vision LLM 场景分析
-> schema 化的导演经验
-> experience cards
-> retrieval_lab 检索与评测
-> 用户端检索、脚本生成、微调和素材辅助
```

## 项目定位

很多 AI 创意 demo 的流程是：

```text
用户 prompt -> LLM -> 一段漂亮文本
```

SceneWeaver 的目标是做成：

```text
真实素材 -> 结构化理解 -> 可检索经验 -> 有证据的生成
```

它的价值不只是“生成脚本”，而是把商业影像中的导演经验拆出来、存下来、检索出来，再作为新项目创作的参考依据。

## 四阶段构建历程

SceneWeaver 不是一次性写成的。它按四个阶段逐步演进，每一阶段都解决一个具体问题，并沉淀出下一阶段可以继续使用的工程资产。

### 第一阶段：镜头拆分

第一阶段解决的是输入问题：真实视频不能直接丢给 LLM，需要先被拆成稳定、可引用、可复查的场景单元。

这一阶段完成了：

- 支持本地视频和 Bilibili URL 输入。
- 使用场景检测把视频切分成 scene span。
- 为每个 scene 抽取 start / middle / end 三帧。
- 可选接入字幕文件，并把字幕片段对齐到 scene。
- 为每个 scene 生成 `ScenePackage`。
- 保留 source、frames、packages 等本地文件结构，方便人工检查。

核心产物：

```text
outputs/film_analysis/<video_id>/
  source/
  frames/
    scene_001_start.jpg
    scene_001_middle.jpg
    scene_001_end.jpg
  packages/
    scene_001.json
    scene_002.json
    scene_packages.json
```

这一阶段的意义是把“视频”变成了“可计算的镜头材料”。从这里开始，后续的分析、检索和生成都能指回具体 scene，而不是停留在模糊的视频整体印象上。

常用命令：

```powershell
python -m sceneweaver.cli package-video "https://www.bilibili.com/video/BVxxxx" --output outputs\film_analysis\BVxxxx
```

本地视频入口：

```powershell
python -m sceneweaver.user_cli ingest path\to\video.mp4 --video-id demo_video --no-analyze
```

### 第二阶段：经验检索

第二阶段解决的是知识复用问题：只分析视频还不够，系统需要把导演经验抽成可检索的单位。

这一阶段完成了：

- 设计 `SceneAnalysis`，让 Vision LLM 输出不只是画面描述，还包括导演解释。
- 从 scene analysis 中抽取 `ExperienceCard`。
- 为经验卡片加入 tags / creative fingerprint。
- 建立半受控 taxonomy，避免检索完全依赖自由文本。
- 支持 keyword brief 到经验卡片的检索。
- 加入 script use case，让卡片知道自己适合脚本中的哪个位置。
- 建立 `mocktesting` 和后来的 `retrieval_lab` 评测思路，开始比较检索策略。

核心产物：

```text
outputs/film_analysis/<video_id>/analysis/
  scene_001.json
  scene_002.json
  scenes.json
  experience_cards.jsonl
  tag_candidates.jsonl
```

`ExperienceCard` 保存的不是普通摘要，而是可复用的导演经验：

```text
source scene ids
+ tags / fingerprint
+ underlying emotion
+ narrative logic
+ director strategy
+ shooting techniques
+ visual symbols
+ copywriting tone
+ reuse condition
+ script use case
+ evidence
```

这一阶段的意义是把“分析结果”变成了“可召回的经验库”。后续用户输入一个创意方向时，系统不是直接让 LLM 发散，而是先找真实案例中的相似表达方式。

常用命令：

```powershell
python -m sceneweaver.cli analyze-scenes outputs\film_analysis\BVxxxx --limit 3 --concurrency 1
python -m sceneweaver.cli extract-experience outputs\film_analysis\BVxxxx
python -m sceneweaver.cli retrieve-cards outputs\film_analysis\BVxxxx "科技向善、可信赖、面对面沟通" --top-k 5
```

带 LLM 意图理解的检索：

```powershell
python -m sceneweaver.cli keyword-loop outputs\film_analysis "科技向善、可信赖、面对面沟通" --intent --top-k 5 --debug
```

### 第三阶段：统一 Schema 并接入 Retrieval Lab

第三阶段解决的是系统整合问题：第一阶段产生真实 scene artifacts，第二阶段产生经验检索逻辑，但两者必须通过统一 schema 接起来，真实输入才能直接进入 retrieval_lab 使用。

这一阶段完成了：

- 统一 `ScenePackage`、`SceneAnalysis`、`ScenesAnalysis`、`TagProfile`、`ExperienceCard` 等核心 schema。
- 让真实视频分析产物和 mock retrieval fixtures 逐步对齐。
- 让 `ExperienceCard` 成为 SceneWeaver 和 Retrieval Lab 之间的稳定接口。
- 在 `retrieval_lab.corpora.sceneweaver` 中适配 SceneWeaver 的 `experience_cards.jsonl`。
- 把 card payload 拆成多个 retrieval channels，用于不同检索策略对比。
- 支持 `summary`、`script_use`、`experience`、`visual_tags`、`tags`、`all` 等 channel policy。
- 让真实输入跑完分析后，可以直接作为 retrieval_lab corpus。
- 增加 retrieval run、index manifest、qrels、metrics、rerank、failure analysis 等实验能力。

第三阶段之后，数据流变成：

```text
真实视频
-> scene packages
-> scene analysis
-> experience_cards.jsonl
-> retrieval_lab index / retrieval run / evaluation
```

这一步很关键。它让项目从“能跑一个端到端 demo”，推进到“真实输入可以进入检索实验平台，被评测、被比较、被迭代”。

Retrieval Lab 可直接读取 SceneWeaver 产物：

```powershell
python -m retrieval_lab index manifest --cards outputs\film_analysis --channel-policy all --output .tmp\retrieval_lab\cards_manifest.json
python -m retrieval_lab retrieval run --cards outputs\film_analysis --query "真实可信的科技向善表达" --channel-policy all --top-k 5
```

常见 channel policy：

```text
summary
script_use
experience
visual_tags
tags
combined
summary_tags
script_experience
all
```

这一阶段的意义是把“项目内部产物”变成了“检索实验平台可消费的数据合同”。这也让后续调参、评测、失败分析不再依赖临时脚本。

### 第四阶段：整理为用户端口与 UI

第四阶段解决的是使用问题：底层 pipeline 跑通之后，需要整理成用户可以操作的入口，而不是只停留在开发者命令集合。

这一阶段完成了：

- 新增稳定用户 API：`sceneweaver.user_api`。
- 新增稳定用户 CLI：`sceneweaver.user_cli`。
- 新增本地 Web UI：`sceneweaver.user_web` + `src/sceneweaver/ui/index.html`。
- UI 支持视频上传、URL 入库、异步 ingest job、日志轮询和取消任务。
- UI 支持检索、参考场景查看、脚本生成和参考素材选择。
- 增加 LLM profile 设置，区分基础文本模型和视觉理解模型。
- 脚本生成支持 creator intent、prompt revision、fine tune instruction、variant generation。
- 增加 script agent：`intent`、`tune`、`assets` 三种模式。

用户端核心 API：

```python
from sceneweaver.user_api import ingest_video, search_scenes, generate_script, run_script_agent_task
```

用户 CLI：

```powershell
python -m sceneweaver.user_cli ingest path\to\video.mp4 --video-id demo_video --limit 3 --concurrency 1
python -m sceneweaver.user_cli search "科技向善、可信赖、面对面沟通" --source outputs\film_analysis --top-k 5
python -m sceneweaver.user_cli script "科技向善、可信赖、面对面沟通" --source outputs\film_analysis --brief "60秒企业招聘宣传片" --duration-seconds 60 --markdown-only
```

启动本地 UI：

```powershell
python -m sceneweaver.user_web --host 127.0.0.1 --port 8765
```

打开：

```text
http://127.0.0.1:8765/
```

第四阶段的意义是把前面三阶段的工程能力收束成用户工作流：入库、检索、选参考、生成脚本、微调方向、推断意图、生成素材建议。

## 当前能力总览

- 本地视频和 Bilibili URL 入库。
- 场景拆分、抽帧、字幕对齐。
- Vision LLM 场景分析。
- Pydantic schema 校验。
- 经验卡片抽取。
- creative fingerprint / tags / taxonomy。
- keyword loop、intent mode、semantic rerank、hybrid retrieval。
- Retrieval Lab 检索实验、qrels、metrics、rerank、failure analysis。
- 用户 CLI 和本地 Web UI。
- LLM provider profile 配置和 ping。
- 基于检索参考的脚本生成。
- creator intent、fine tune instruction、prompt revision、variant generation。
- script agent 的 intent / tune / assets 三类辅助接口。

## 当前状态

SceneWeaver 现在是一个可运行的本地 AI 创意工作流原型。它已经有端到端路径、真实输入接入、经验检索、retrieval_lab 对接、用户端 API 和本地 UI。

它还不是生产级 SaaS。当前设计优先保证：

- 产物可检查；
- schema 可验证；
- 检索可解释；
- 本地流程可复现；
- 后续客制化容易扩展。

本地测试结果：

```text
301 passed
```

## 安装

需要 Python 3.11+。

基础开发环境：

```powershell
python -m pip install -e ".[dev,video]"
```

可选语义检索：

```powershell
python -m pip install -e ".[semantic]"
```

真实视频处理还需要本机可用的 `ffmpeg`。`video` extra 会安装 Python 侧依赖，例如 `yt-dlp`、`scenedetect[opencv]`、`httpx`，但不会安装系统级 ffmpeg。

## LLM 配置

SceneWeaver 使用 OpenAI-compatible chat/completions API。

DashScope / 百炼兼容配置示例：

```powershell
$env:DASHSCOPE_API_KEY="sk-..."
$env:DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:DASHSCOPE_MODEL="qwen3.7-max"
```

通用配置：

```powershell
$env:SCENEWEAVER_API_KEY="sk-..."
$env:SCENEWEAVER_BASE_URL="https://api.example.com/v1"
$env:SCENEWEAVER_MODEL="your-model-id"
```

诊断命令：

```powershell
python -m sceneweaver.cli llm-status --provider auto --include-models
python -m sceneweaver.cli llm-check "hi" --timeout-seconds 30
```

本地 UI 也可以保存不同模型 profile，例如文本生成模型和视觉理解模型。配置文件默认保存在：

```text
.tmp/sceneweaver/llm_config.json
```

## 快速开始

### 1. 本地 smoke test

不调用外部 API：

```powershell
python -m sceneweaver.cli mock-run --output outputs\mock\quick_check
```

预期产物：

```text
outputs/mock/quick_check/analysis/experience_cards.jsonl
```

### 2. 启动 UI

```powershell
python -m sceneweaver.user_web --host 127.0.0.1 --port 8765
```

或者安装后使用脚本入口：

```powershell
sceneweaver-ui --host 127.0.0.1 --port 8765
```

### 3. 端到端运行真实视频

```powershell
python -m sceneweaver.cli run "https://www.bilibili.com/video/BVxxxx" --limit 3 --concurrency 1
```

分步运行：

```powershell
python -m sceneweaver.cli package-video "https://www.bilibili.com/video/BVxxxx" --output outputs\film_analysis\BVxxxx
python -m sceneweaver.cli analyze-scenes outputs\film_analysis\BVxxxx --limit 3 --concurrency 1
python -m sceneweaver.cli extract-experience outputs\film_analysis\BVxxxx
```

### 4. 检索经验卡片

```powershell
python -m sceneweaver.user_cli search "科技向善、可信赖、面对面沟通" --source outputs\film_analysis --top-k 5
```

### 5. 基于参考生成脚本

```powershell
python -m sceneweaver.user_cli script "科技向善、可信赖、面对面沟通" --source outputs\film_analysis --brief "60秒企业招聘宣传片" --duration-seconds 60 --tone "克制、真实、有信任感" --audience "年轻工程师" --top-k 5 --markdown-only
```

## 用户端 API

`src/sceneweaver/user_api.py` 是目前整理给用户端和 UI 调用的稳定接口层。

```python
from sceneweaver.user_api import search_scenes, generate_script, run_script_agent_task

search = search_scenes(
    "科技向善、可信赖、面对面沟通",
    ["outputs/film_analysis"],
    top_k=5,
)

script = generate_script(
    "科技向善、可信赖、面对面沟通",
    ["outputs/film_analysis"],
    script_brief="60秒企业招聘宣传片",
    duration_seconds=60,
    tone="克制、真实、有信任感",
    audience="年轻工程师",
    creator_intent_prompt="希望更像真实对话，不要口号感。",
    fine_tune_instruction="减少宏大叙事，多用具体工作瞬间。",
    prompt_revision_rounds=1,
    variant_count=2,
)

intent = run_script_agent_task(
    "intent",
    context={"script": script.get("script")},
    user_input="我想让它更真诚一点，帮我推断我的偏好。",
)
```

Script agent 支持：

```text
intent  推断创作者真实意图、偏好和追问
 tune   给当前脚本提出微调方向和 instruction
assets  根据脚本和参考生成素材、拍摄、文案等辅助建议
```

## 本地 UI HTTP 接口

`sceneweaver.user_web` 提供一个本地 HTTP server，服务 `src/sceneweaver/ui/index.html`。

主要接口：

```text
GET  /api/status
GET  /api/sources
GET  /api/llm-settings
POST /api/llm-settings
POST /api/llm-ping
POST /api/ingest
POST /api/ingest-jobs
POST /api/upload-ingest
POST /api/upload-ingest-jobs
POST /api/search
POST /api/generate-script
POST /api/script-agent
```

异步 ingest job 接口用于长视频分析，UI 可以轮询日志、查看状态和取消任务。

## 目录结构

```text
src/sceneweaver/
  analysis/          场景分析、标签、fingerprint、经验卡片、keyword loop
  input/             Bilibili、字幕和输入处理
  llm/               OpenAI-compatible LLM client、provider、profile 设置
  pipeline/          视频打包、本地视频 pipeline、mock pipeline
  retrieval/         经验卡片检索、script use case、rerank 辅助
  schemas/           核心 Pydantic schema
  split/             场景检测、抽帧、字幕分段、timecode
  ui/                本地浏览器 UI
  user_api.py        用户端稳定 API
  user_cli.py        用户端 CLI
  user_web.py        本地 UI server

src/retrieval_lab/
  datasets/          数据集和 fixture 工具
  planners/          query planner 和 planner cache
  indexes/           index manifest 和 card channel policy
  retrieval/         retrieval runtime 和命令层
  ranking/           rerank 与诊断 workflow
  qrels/             graded relevance 和 review workflow
  evaluators/        IR metrics 和 failure analysis
  experiments/       实验编排和报告

src/mocktesting/     历史检索 fixture 和 baseline 工具
tests/               单元测试、CLI 测试、检索测试、UI 测试
docs/                架构、CLI、用法、schema、路线图
prompts/             场景分析、联想、导演脚本生成 prompt
taxonomy/            半受控导演标签体系
outputs/             本地生成产物
```

## 关键 Schema

主要 schema 位于 `src/sceneweaver/schemas/`。

```text
ScenePackage       镜头拆分后的场景包
SceneAnalysis      视觉观察和导演解释
ScenesAnalysis     单个视频的场景分析集合
TagProfile         tags / creative fingerprint
ExperienceCard     可复用导演经验
ScriptUseCase      卡片适合脚本中的哪个位置
FilmAnalysis       全片分析
AssociationAnalysis 创意联想输出
CreativeIntentAnalysis 检索用的意图理解
```

当前 fingerprint 维度：

```text
emotion_core
audience_projection
narrative_function
interaction_mode
visual_motifs
symbolic_logic
rhythm_pattern
```

## 输出产物

典型真实视频输出：

```text
outputs/film_analysis/<video_id>/
  source/
  frames/
  packages/
    scene_001.json
    scene_002.json
    scene_packages.json
  analysis/
    scene_001.json
    scene_002.json
    scenes.json
    film_analysis.json
    experience_cards.jsonl
    tag_candidates.jsonl
```

其中：

- `packages/*.json` 是镜头拆分结果。
- `analysis/scenes.json` 是 Vision LLM 的结构化场景分析集合。
- `analysis/experience_cards.jsonl` 是后续检索和生成主要使用的经验库。
- `analysis/tag_candidates.jsonl` 记录未纳入 taxonomy 的候选标签。

## 测试

运行全部测试：

```powershell
python -m pytest -q
```

聚焦检索：

```powershell
python -m pytest tests\test_retrieval_service.py tests\test_retrieval_usecase.py tests\test_keyword_loop.py -q
```

聚焦用户端和 UI：

```powershell
python -m pytest tests\test_user_cli.py tests\test_user_web.py -q
```

如果 Windows 临时目录权限影响 pytest，可以把临时目录放到项目内：

```powershell
$env:TMP="D:\WorkSpace\Codex\SceneWeaver\.tmp"
$env:TEMP="D:\WorkSpace\Codex\SceneWeaver\.tmp"
python -m pytest -q
```

## 设计原则

- 先用本地 JSON / JSONL 保存关键产物，保证可检查。
- LLM 输出必须经过 schema 校验，不直接信任模型文本。
- 检索结果必须尽量能指回真实 scene evidence。
- 高密度解释用于人读，低维 tags / fingerprint 用于检索。
- 先把小规模 in-memory 检索和评测做扎实，再考虑 vector database。
- 用户端口只包装稳定能力，不把实验性内部细节直接暴露给用户。

## 当前限制

- 当前项目偏本地工作流，还不是多用户生产部署。
- 真实视频处理依赖 ffmpeg、网络状态和视频平台可访问性。
- Vision 分析质量依赖所选多模态模型。
- 部分历史 fixture 或旧产物可能仍有编码污染。
- 当前默认不使用 vector database，小规模经验库优先保持可解释和易调试。
- full-film analysis 和 experience extraction 还可以继续增强，使全片理解更稳定地约束卡片生成。

## 后续方向

- 强化 full-film analysis，让它更好地约束 experience card extraction。
- 把 creative fingerprint 从兼容层推进为更明确的中间产物。
- 增加 tag candidate review / merge / reject 工作流。
- 扩充真实视频验证集和 retrieval_lab 评测报告。
- 优化 UI 中的参考选择、脚本版本对比和素材生成体验。
- 加入更清晰的成本、延迟、模型 provider 诊断。
- 在数据规模真正变大时，再评估 vector database 或生产数据库。

## 文档

- [Documentation Index](docs/README.md)
- [CLI Command Book](docs/CLI.md)
- [Usage Guide](docs/USAGE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Schema](docs/SCHEMA.md)
- [Roadmap](docs/ROADMAP.md)
- [Retrieval Lab](src/retrieval_lab/README.md)

## License

MIT. See [LICENSE](LICENSE).
