# SceneWeaver

SceneWeaver 是一个面向商业宣传片、招聘宣传片和品牌短片的导演经验分析系统。

它不直接生成视频，而是把已有视频中的导演语言拆解成结构化数据，并为后续的经验检索、导演稿生成和创意联想提供知识基础。

## 当前状态

截至 2026-05-09，项目处于 v1 中段：

```text
已完成：工程骨架、schema、mock pipeline、真实视频 package pipeline、scene-level Vision LLM 分析、creative fingerprint 中间层
已验收：真实视频 40 个 scene 的 package、scene analysis 和 fingerprint 生成
待实现：字幕自动获取、full-film analysis、experience card 自动抽取、experience card 检索入口、完整 v1 后半段闭环
```

当前判断：

```text
真实视频到 scene-level director analysis 和 fingerprint corpus 的前半段已经成立。
full-film analysis、experience cards 和 retrieval 仍是 v1 后半段。
```

## v1 目标

v1 要跑通“视频到导演经验”的数据生产线：

```text
Bilibili URL
→ 下载视频和 metadata
→ scene 检测
→ start / middle / end 三帧抽取
→ scene package
→ scene-level Vision LLM 分析
→ scenes.json
→ creative fingerprints
→ full-film LLM 分析
→ film_analysis.json
→ experience card 抽取
→ experience_cards.jsonl
```

v1 成功标准不是“生成一份好看的报告”，而是：

```text
能否稳定提取出可复用、可验证、可检索的导演经验卡片。
```

## 已具备能力

1. `mock-run`：生成完整 mock 产物并通过 schema validation。
2. `package-video`：下载 Bilibili 视频，检测 scene，抽三帧，生成 scene packages。
3. `analyze-scenes`：读取 scene packages 和三帧图，调用 Vision LLM 生成 scene analysis。
4. `fingerprint-scenes`：从 scene analysis 生成 scene / film creative fingerprints。
5. `associate`：把关键词或粗糙 brief 扩展为导演/编剧可用的联想材料，并补 query fingerprint。
6. `run`：串起真实视频 package、scene analysis 和 creative fingerprint generation。

已完成真实 package 样本：

```text
BV1pLqnBWEJC
scene_count: 16
frame_count: 48
package_count: 16
```

已完成真实 scene analysis + fingerprint 样本：

```text
BV1cWHyzwEKC
command: python -m sceneweaver.cli run "https://www.bilibili.com/video/BV1cWHyzwEKC" --limit 40 --concurrency 5
scenes analyzed: 40
scenes fingerprinted: 40
outputs:
  outputs/film_analysis/BV1cWHyzwEKC/analysis/scenes.json
  outputs/film_analysis/BV1cWHyzwEKC/fingerprints/film_fingerprint.json
```

## 快速开始

推荐使用 Python 3.11 环境。当前仓库尚未完成干净环境的一键可复现配置，后续需要补依赖锁定和 CLI smoke test。

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
python -m sceneweaver.cli package-video "https://www.bilibili.com/video/BV1pLqnBWEJC" --output outputs\film_analysis\BV1pLqnBWEJC
```

scene 级 LLM 分析：

```powershell
$env:SCENEWEAVER_API_KEY="..."
$env:SCENEWEAVER_BASE_URL="https://..."
$env:SCENEWEAVER_MODEL="..."

python -m sceneweaver.cli analyze-scenes outputs\film_analysis\BV1pLqnBWEJC --limit 1 --concurrency 1
```

生成 creative fingerprints：

```powershell
python -m sceneweaver.cli fingerprint-scenes outputs\film_analysis\BV1pLqnBWEJC
```

真实视频端到端跑到 fingerprint：

```powershell
python -m sceneweaver.cli run "https://www.bilibili.com/video/BV1cWHyzwEKC" --limit 40 --concurrency 5
```

关键词联想：

```powershell
python -m sceneweaver.cli associate "青春 / 逆光 / 奔跑 / 创意 / 不惧挑战"
python -m sceneweaver.cli associate "招聘宣传片 / 科技向善 / 提供机会发挥潜力" --debug --timeout-seconds 240 --retries 2
python -m sceneweaver.cli associate "招聘宣传片 / 科技向善 / 提供机会发挥潜力" --stream
```

## 输出目录

```text
outputs/
  film_analysis/<BV号>/
    source/
    frames/
    packages/
    analysis/
    fingerprints/
  key_associates/
  mock/
```

主要产物：

1. `packages/scene_XXX.json`：送入 Vision LLM 的 scene package。
2. `analysis/scene_XXX.json`：单个 scene 的导演语言分析。
3. `analysis/scenes.json`：全片 scene analysis 汇总。
4. `fingerprints/scene_XXX.json`：单个 scene 的低维语义坐标。
5. `fingerprints/film_fingerprint.json`：全片 fingerprint 聚合。
6. `analysis/film_analysis.json`：全片导演语言总结，待实现真实链路。
7. `analysis/experience_cards.jsonl`：可复用导演经验卡片，待实现真实链路。

当前真实 `run` 命令已经稳定产出到第 5 项。第 6、7 项是下一阶段核心工作。

## 文档分工

为避免文档互相重复，后续按下面分工维护：

1. [开发计划](docs/PLAN.md)：当前 v1 执行计划和下一步顺序。
2. [执行状态](docs/EXECUTION_STATUS.md)：当前完成度、验收记录和已知环境问题。
3. [技术设计](docs/TECHNICAL_DESIGN.md)：代码架构、模块职责和运行链路。
4. [数据结构](docs/SCHEMA.md)：核心 JSON / JSONL schema。
5. [产品需求](docs/PRD.md)：产品目标、用户场景和非目标。
6. [路线图](docs/ROADMAP.md)：v2 以后方向。
7. [背景摘要](docs/CONTEXT_SUMMARY.md)：交接用短摘要。
8. [开发日志](docs/DEVELOPMENT_LOG.md)：按时间记录已发生的工程推进。
9. [参考项目笔记](docs/REFERENCE_NOTES.md)：本地参考项目的借鉴边界。

## 核心原则

1. 先结构化，再自动化。
2. 先本地文件，再数据库。
3. 先小样本验证，再大规模处理。
4. 所有 LLM 输出必须通过 Pydantic validation。
5. `visual_observation` 写客观观察，`director_interpretation` 写推断和解释，`experience_card` 写可复用经验。
