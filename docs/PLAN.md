# 开发计划

本文是当前执行计划。完成度见 `docs/EXECUTION_STATUS.md`，长期方向见 `docs/ROADMAP.md`。

## 1. 当前目标

v1 的目标是跑通“真实视频到导演经验卡片”的最小闭环：

```text
Bilibili URL
→ scene packages
→ scene analysis
→ film analysis
→ experience cards
```

当前判断：

```text
前半段已经用真实视频跑通到 scene analysis + creative fingerprints。
后半段需要继续实现 film analysis、experience cards 和 retrieval。
```

## 2. 当前状态

已完成：

1. 工程骨架、schema、mock pipeline。
2. 真实 Bilibili 视频下载和 scene package 生成。
3. start / middle / end 三帧抽取。
4. scene-level Vision LLM 分析代码和真实 40 scene 样本验收。
5. CreativeFingerprint schema、scene fingerprint、film fingerprint 和 query fingerprint。
6. `associate` 关键词联想命令。
7. `run` 命令已串起 package、scene analysis、fingerprint generation。

待完成：

1. 环境可复现和 CLI help 稳定性。
2. Bilibili 字幕自动获取。
3. full-film analysis。
4. experience card extraction。
5. query fingerprint 到 experience cards 的检索入口。
6. `run` 命令完整 v1 后半段闭环。

## 3. P0：先让当前代码可稳定验收

### 3.1 修环境可复现

目标：

```text
干净环境可以安装、运行 CLI、执行测试。
```

任务：

1. 确认推荐 Python 版本为 3.11。
2. 调整 `pyproject.toml` 依赖版本，解决 Typer / Click `--help` 报错。
3. 补充安装命令和开发环境说明。
4. 增加 CLI `--help` smoke test。
5. 重新执行 `python -m pytest`。

验收：

```powershell
python -m sceneweaver.cli --help
python -m sceneweaver.cli mock-run --output outputs\mock\quick_check
python -m pytest
```

### 3.2 维护真实 scene-level LLM 验收

目标：

```text
把已经跑通的真实 Vision LLM 样本变成可复查的验收基线。
```

已完成样本：

```powershell
python -m sceneweaver.cli run "https://www.bilibili.com/video/BV1cWHyzwEKC" --limit 40 --concurrency 5
```

产物：

```text
outputs/film_analysis/BV1cWHyzwEKC/analysis/scenes.json
outputs/film_analysis/BV1cWHyzwEKC/fingerprints/film_fingerprint.json
```

后续维护任务：

1. 保留该样本作为 scene analysis 和 fingerprint 的回归基线。
2. 抽查 `analysis/scene_XXX.json` 内容质量。
3. 必要时调整 `prompts/scene_analysis.md`。
4. 记录 provider、model 和运行参数，便于复现。

回归命令：

```powershell
python -m sceneweaver.cli run "https://www.bilibili.com/video/BV1cWHyzwEKC" --limit 40 --concurrency 5
```

验收标准：

1. 输出 `analysis/scene_001.json`。
2. 输出通过 `SceneAnalysis` validation。
3. `visual_observation` 不混入导演推断。
4. `director_interpretation` 基于画面和字幕，不伪造事实。
5. 不出现评分、筛选等级、`weighted_score`。

## 4. P1：补齐 v1 后半段

### 4.1 Full-film analysis

目标：

```text
从 scenes.json 生成全片导演语言总结。
```

计划新增：

1. `src/sceneweaver/analysis/film_analyzer.py`
2. `prompts/film_analysis.md`
3. CLI `analyze-film`

输入：

```text
analysis/scenes.json
```

输出：

```text
analysis/film_analysis.json
```

验收标准：

1. 输出通过 `FilmAnalysis` validation。
2. 基于 scene 时间顺序分析全片。
3. 能总结氛围、节奏、情绪曲线、视觉语言、叙事结构、品牌人格。

### 4.2 Experience card extraction

目标：

```text
从 scene analysis 和 film analysis 抽取可复用导演经验。
```

计划新增：

1. `src/sceneweaver/analysis/experience_extractor.py`
2. `prompts/experience_extraction.md`
3. CLI `extract-experience`

输入：

```text
analysis/scenes.json
analysis/film_analysis.json
```

输出：

```text
analysis/experience_cards.jsonl
```

验收标准：

1. 每张 card 通过 `ExperienceCard` validation。
2. 每张 card 保留来源 video 和 scene ids。
3. 内容不是 scene 摘要，而是可迁移的导演经验。
4. 包含底层情绪、叙事逻辑、导演策略、拍摄技法、视觉符号和复用条件。

### 4.3 完整 run 闭环

目标：

```text
run 命令产出 v1 全部核心文件。
```

目标链路：

```text
package-video
→ analyze-scenes
→ fingerprint-scenes
→ analyze-film
→ extract-experience
```

验收命令：

```powershell
python -m sceneweaver.cli run "https://www.bilibili.com/video/BV1pLqnBWEJC" --limit 20 --concurrency 3
```

验收产物：

1. `packages/scene_XXX.json`
2. `packages/scene_packages.json`
3. `analysis/scene_XXX.json`
4. `analysis/scenes.json`
5. `fingerprints/scene_XXX.json`
6. `fingerprints/film_fingerprint.json`
7. `analysis/film_analysis.json`
8. `analysis/experience_cards.jsonl`

### 4.4 Experience card retrieval

目标：

```text
用 query_fingerprint 召回 grounded experience cards，而不是只做自由联想。
```

计划新增：

1. CLI `retrieve-cards`
2. 从 brief 生成 `query_fingerprint`
3. 读取 `analysis/experience_cards.jsonl`
4. 输出 top-k cards 和匹配得分

验收标准：

1. 输入腾讯 2019 宣传片式 brief 时，能优先召回 `direct_address`、`screen`、`trust`、`human_centered_technology` 等相关 cards。
2. 每个召回结果能展示 fingerprint overlap 和 evidence。

## 5. P2：提升输入质量

### 5.1 字幕自动获取

目标：

```text
有字幕时自动注入 scene package；无字幕时不阻塞 pipeline。
```

任务：

1. 调研 `yt-dlp` Bilibili 字幕输出。
2. 保存 `source/subtitles.srt`。
3. 在 package 阶段默认读取字幕。
4. 将字幕来源和失败原因写入 metadata。

### 5.2 LLM 请求可靠性

任务：

1. 给 `analyze_images_json` 增加 retry / timeout。
2. 统一文本和图像请求的 provider 错误提示。
3. 处理 partial JSON 和 schema validation failure。
4. 保证失败 scene 不阻塞已有成功结果。

## 6. 当前命令大纲

mock：

```powershell
python -m sceneweaver.cli mock-run --output outputs\mock\quick_check
```

真实视频 package：

```powershell
python -m sceneweaver.cli package-video "https://www.bilibili.com/video/BV1pLqnBWEJC" --output outputs\film_analysis\BV1pLqnBWEJC
```

scene 分析：

```powershell
python -m sceneweaver.cli analyze-scenes outputs\film_analysis\BV1pLqnBWEJC --limit 20 --concurrency 3
```

关键词联想：

```powershell
python -m sceneweaver.cli associate "青春 / 逆光 / 奔跑 / 创意 / 不惧挑战"
```

## 7. 开发原则

1. 先结构化，再自动化。
2. 先本地文件，再数据库。
3. 先小样本验证，再大规模处理。
4. 所有 LLM 输出必须通过 Pydantic validation。
5. 所有中间结果都要可落盘、可复跑、可检查。
