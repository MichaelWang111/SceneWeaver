# SceneWeaver

SceneWeaver 是一个从商业视频、招聘宣传片和品牌短片中提炼导演经验的本地 Python 项目。

它不直接生成视频，而是把已有视频拆成可检查的 scene package，再产出带半封闭 tags 的 scene analysis，并从中沉淀 experience cards。当前重构目标是压缩核心解析架构，避免 schema boom。

## 当前状态

截至 2026-05-11，项目正在从旧的 `fingerprints/` 独立中间层迁移到：

```text
analysis 内嵌 tags
→ experience_cards.jsonl
```

已经成立：

1. 真实视频 package pipeline。
2. scene-level Vision LLM analysis。
3. `SceneAnalysis.tags` 第一版生成。
4. `ExperienceCard.tags` 第一版检索。
5. `associate` 作为独立创意联想工具保留。

正在收敛：

1. 停止把 tags 写入独立 `fingerprints/` 目录。
2. `fingerprint-scenes` 只作为 legacy 过渡命令，用于给旧 analysis 补 tags。
3. `FilmAnalysis` 暂停作为核心链路，后续只作为报告型派生产物。

## 核心架构

核心输出目录只保留：

```text
outputs/film_analysis/<video_id>/
  source/
  frames/
  packages/
  analysis/
```

核心语义层只有两层：

```text
SceneAnalysis.tags
ExperienceCard.tags
```

`tags` 是半封闭标签集：正式数据只写 canonical tags；新表达、同义词、近义词先进入 aliases 或 candidate pool，再决定合并、升级或丢弃。

## v1 链路

```text
Bilibili URL
→ 下载视频和 metadata
→ scene 检测
→ start / middle / end 三帧抽取
→ scene package
→ scene-level Vision LLM analysis
→ analysis/scene_XXX.json with tags
→ analysis/scenes.json
→ analysis/experience_cards.jsonl
```

## 命令

安装：

```powershell
python -m pip install -e ".[dev,video]"
```

生成 mock 产物：

```powershell
python -m sceneweaver.cli mock-run --output outputs\mock\quick_check
```

真实视频打包：

```powershell
python -m sceneweaver.cli package-video "https://www.bilibili.com/video/BVxxxx" --output outputs\film_analysis\BVxxxx
```

scene 分析，输出会包含 `tags`：

```powershell
python -m sceneweaver.cli analyze-scenes outputs\film_analysis\BVxxxx --limit 20 --concurrency 3
```

legacy 过渡命令：给旧 analysis 补 tags，不再写 `fingerprints/`：

```powershell
python -m sceneweaver.cli fingerprint-scenes outputs\film_analysis\BVxxxx
```

抽取经验卡片：

```powershell
python -m sceneweaver.cli extract-experience outputs\film_analysis\BVxxxx
```

检索经验卡片：

```powershell
python -m sceneweaver.cli retrieve-cards outputs\film_analysis\BVxxxx "招聘宣传片，稳重可靠，科技向善，面对屏幕后的观众对话" --top-k 3
```

端到端运行：

```powershell
python -m sceneweaver.cli run "https://www.bilibili.com/video/BVxxxx" --limit 40 --concurrency 5
```

关键词联想：

```powershell
python -m sceneweaver.cli associate "青春 / 逆光 / 奔跑 / 创意 / 不惧挑战"
```

## 核心产物

1. `packages/scene_XXX.json`：送入 Vision LLM 的 scene 输入包。
2. `analysis/scene_XXX.json`：单个 scene 的导演分析，内含 `tags`。
3. `analysis/scenes.json`：全片 scene analysis 汇总。
4. `analysis/experience_cards.jsonl`：可复用导演经验卡片，内含 `tags`。
5. `analysis/film_analysis.json`：legacy/mock 报告型产物，暂不作为核心链路。

## 核心原则

1. 先结构化，再自动化。
2. 先本地文件，再数据库。
3. 正式数据只写 canonical tags。
4. 新表达先进入 candidate pool，不直接污染主标签集。
5. `visual_observation` 写客观观察，`director_interpretation` 写推断解释，`experience_cards` 写可复用经验。
6. 所有 LLM 输出必须通过 Pydantic validation。
