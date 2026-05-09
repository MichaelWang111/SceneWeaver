# 执行状态

本文只记录当前完成度、验收状态和已知问题。开发计划见 `docs/PLAN.md`，长期路线见 `docs/ROADMAP.md`。

## 1. 当前结论

更新时间：2026-05-09

```text
真实视频到 scene analysis + creative fingerprint 的前半段已经跑通。
v1 后半段 film analysis、experience cards 和 retrieval 尚未完整闭环。
```

已经成立：

1. 项目方向和 v1 范围明确。
2. Python 工程骨架、schema、mock pipeline 已实现。
3. 真实 Bilibili 视频 package pipeline 已完成样本验收。
4. scene-level Vision LLM 分析已完成真实 40 scene 样本验收。
5. CreativeFingerprint 中间层已接入真实 `run` 链路。
6. 关键词联想 `associate` 已实现，并支持 query fingerprint。

尚未成立：

1. 干净环境一键安装和测试验证。
2. Bilibili 字幕自动获取。
3. full-film analysis 真实链路。
4. experience card 自动抽取真实链路。
5. query fingerprint 到 experience cards 的检索入口。
6. `run` 命令完整产出 `scenes.json -> fingerprints -> film_analysis.json -> experience_cards.jsonl`。

## 2. 状态表

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| 文档和方向 | 已完成，已整理 | README 作为入口，PLAN/STATUS/ROADMAP 分工维护 |
| Python 包结构 | 已完成 | `src/sceneweaver/` |
| Schema | 已完成基础版 | Pydantic models 和 example validation |
| Mock pipeline | 已完成 | 可生成 package、scene analysis、film analysis、cards |
| Bilibili 下载 | 已实现 | 基于 `yt-dlp` |
| Scene 检测 | 已实现并验收 | 基于 PySceneDetect |
| 三帧抽取 | 已实现并验收 | 基于 ffmpeg |
| SRT 字幕切片 | 已实现 | 需要用户提供字幕文件 |
| 字幕自动获取 | 未实现 | v1 输入质量的下一项补强 |
| Scene LLM 分析 | 已完成真实样本验收 | `BV1cWHyzwEKC` 已分析 40 个 scene |
| Creative Fingerprint | 已接入真实链路 | scene fingerprints 和 film_fingerprint 已产出 |
| Full-film 分析 | 待实现 | schema/mock 已有，真实链路未接 |
| Experience card 抽取 | 待实现 | schema/mock 已有，真实链路未接 |
| Experience card 检索 | 待实现入口 | 底层 fingerprint scoring 已有，CLI/API 待接 |
| 关键词联想 | 已实现 | `associate` 命令，文本 LLM JSON 输出，支持 query fingerprint |

## 3. 已验收样本

真实 package pipeline 样本：

```text
BV1pLqnBWEJC
标题：智变之时 | 2025腾讯ConTech大会开场视频
scene_count: 16
frame_count: 48
package_count: 16
manifest: packages/scene_packages.json
```

命令：

```powershell
python -m sceneweaver.cli package-video "https://www.bilibili.com/video/BV1pLqnBWEJC" --output outputs\film_analysis\BV1pLqnBWEJC
```

说明：

1. `scenes/` 目录默认不生成 scene mp4 clips。
2. 需要 clips 时使用 `--split-video`。
3. 当前样本未自动获取字幕。

真实 scene analysis + fingerprint 样本：

```text
BV1cWHyzwEKC
命令：python -m sceneweaver.cli run "https://www.bilibili.com/video/BV1cWHyzwEKC" --limit 40 --concurrency 5
Scenes analyzed: 40
Scenes fingerprinted: 40
```

核心产物：

```text
outputs/film_analysis/BV1cWHyzwEKC/packages/
outputs/film_analysis/BV1cWHyzwEKC/analysis/scenes.json
outputs/film_analysis/BV1cWHyzwEKC/fingerprints/film_fingerprint.json
```

说明：

1. 该样本证明真实 Vision LLM scene analysis 和 fingerprint generation 已能连续运行。
2. 该目录尚未产出 `analysis/film_analysis.json`。
3. 该目录尚未产出 `analysis/experience_cards.jsonl`。

## 4. 测试和环境记录

历史记录中曾在项目环境跑通过不同阶段测试：

```text
v1-0: pytest 8 passed
v1-1: pytest 15 passed
v1-2 scene-level mocked tests: pytest 17 passed
当前测试文件数量：34 个 test case
```

本次整理时观察到当前 shell 环境存在问题：

1. 当前 Python 是 `3.13.5`，不是推荐的 Python 3.11 环境。
2. 当前环境没有 `pytest`。
3. 当前环境没有 `openai`。
4. 当前环境未安装 editable package，直接 `python -m sceneweaver.cli` 找不到包。
5. 临时设置 `PYTHONPATH=src` 后，`mock-run` 可以执行成功。
6. 临时设置 `PYTHONPATH=src` 后，`--help` 在 `typer 0.9.0 + click 8.2.1` 下报错。

需要补的工程项：

1. 锁定或调整 Typer / Click 版本。
2. 补一个干净环境安装说明。
3. 增加 CLI `--help` smoke test。
4. 重新执行 `python -m pytest` 并更新本文件。

## 5. 下一步验收

优先顺序：

1. 修环境和依赖版本，确保 CLI 和测试可复现。
2. 抽查 `BV1cWHyzwEKC` 的 scene analysis 和 fingerprint 质量，记录为回归基线。
3. 实现 `film_analyzer`，从 `analysis/scenes.json` 和 `fingerprints/film_fingerprint.json` 生成 `analysis/film_analysis.json`。
4. 实现 `experience_extractor`，生成带 fingerprint 的 `analysis/experience_cards.jsonl`。
5. 实现 query fingerprint 到 experience cards 的 top-k retrieval 入口。
