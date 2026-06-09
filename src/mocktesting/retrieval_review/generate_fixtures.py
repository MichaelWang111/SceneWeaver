from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


SCENE_PLAN = [
    ("scene_001", "opening", "开场建立真实世界压力", ["opening", "establish_problem", "build_reality"]),
    ("scene_002", "setup", "进入一线场景，建立具体需求", ["setup", "establish_need", "show_pressure"]),
    ("scene_003", "character_intro", "引入核心人物和专业视角", ["character_intro", "introduce_people", "build_empathy"]),
    ("scene_004", "team_work", "展现跨职能团队如何协作", ["team_work", "show_team", "show_collaboration"]),
    ("scene_005", "technology_showcase", "让技术或系统能力自然入场", ["technology_showcase", "show_technology", "prove_capability"]),
    ("scene_006", "scale_reveal", "揭示世界级规模和网络能力", ["scale_reveal", "show_scale", "show_network"]),
    ("scene_007", "value_expression", "通过具体互动建立信任和价值", ["value_expression", "build_trust", "show_responsibility"]),
    ("scene_008", "outcome", "呈现结果落地但避免夸大", ["outcome", "show_outcome", "land_value"]),
    ("scene_009", "growth", "表现员工、客户或系统的成长", ["growth", "show_growth", "show_long_termism"]),
    ("scene_010", "ending", "收束主题并完成品牌表达", ["ending", "close_loop", "express_value"]),
]


FIXTURES = [
    {
        "fixture_id": "enterprise_healthtech_documentary_002",
        "title": "每一次确认都算数",
        "industry": "medical_technology_and_digital_health",
        "style": "documentary_enterprise_brand_film",
        "duration_seconds": 180,
        "logline": "跟随三位一线医生、质控工程师和患者服务专员，记录医疗设备从校准、使用到回访的完整信任链路。",
        "brand_personality": ["严谨", "克制", "可信赖", "专业", "长期陪伴"],
        "world": {
            "place": "城市三甲医院、设备质控实验室、患者随访中心",
            "hero": "医疗设备质控工程师",
            "problem": "医疗判断需要稳定、可追溯、被反复确认的技术支持",
            "technology": "影像设备质控系统和远程维护平台",
            "team": "医生、质控工程师、服务工程师、患者随访专员",
            "outcome": "医生获得稳定设备状态，患者获得清晰解释和持续随访",
            "symbol": "校准线、质控标签、复核记录"
        },
    },
    {
        "fixture_id": "enterprise_healthtech_short_003",
        "title": "30秒里的安心",
        "industry": "medical_technology_and_digital_health",
        "style": "short_video_vertical_world_500_style",
        "duration_seconds": 75,
        "logline": "用高密度短片节奏串联急诊、平台、专家连线和患者安心瞬间，做一支不浮夸的医疗科技竖屏片。",
        "brand_personality": ["敏捷", "温暖", "可靠", "清晰", "人本"],
        "world": {
            "place": "急诊走廊、移动端会诊界面、医生办公室、病房",
            "hero": "年轻急诊医生",
            "problem": "关键时刻需要快速连接可信医学资源",
            "technology": "移动远程会诊和AI预筛系统",
            "team": "急诊医生、远程专家、产品响应团队",
            "outcome": "复杂病例快速进入清晰处理路径",
            "symbol": "倒计时、手机震动、确认按钮"
        },
    },
    {
        "fixture_id": "enterprise_energy_business_001",
        "title": "让电流更稳地抵达",
        "industry": "energy_and_utilities",
        "style": "business_corporate_brand_film",
        "duration_seconds": 160,
        "logline": "从城市调度中心到海上风电场，展示能源企业如何用数字化系统保障稳定供给和低碳转型。",
        "brand_personality": ["稳健", "可靠", "有规模", "负责任", "面向未来"],
        "world": {
            "place": "城市电网调度中心、海上风电场、储能站、制造基地",
            "hero": "电网调度负责人",
            "problem": "城市用电高峰与新能源波动需要被稳定协调",
            "technology": "智能电网调度平台和储能预测系统",
            "team": "调度员、风电工程师、储能运维团队、数据科学家",
            "outcome": "新能源更稳定地进入城市生活和产业生产",
            "symbol": "电网曲线、海风、储能柜指示灯"
        },
    },
    {
        "fixture_id": "enterprise_energy_documentary_002",
        "title": "风经过的地方",
        "industry": "energy_and_utilities",
        "style": "documentary_enterprise_brand_film",
        "duration_seconds": 190,
        "logline": "记录风电运维团队在恶劣天气窗口中完成检修，呈现低碳能源背后的专业和耐心。",
        "brand_personality": ["坚韧", "朴素", "专业", "环保", "长期主义"],
        "world": {
            "place": "沿海小镇、海上风电平台、运维船、远程监测室",
            "hero": "海上风电运维工程师",
            "problem": "清洁能源不是口号，而是日复一日的安全维护和可靠交付",
            "technology": "风机预测性维护系统和海况监测平台",
            "team": "运维工程师、安全员、数据监测员、港口协调员",
            "outcome": "检修完成，风机重新并网，沿海工厂恢复稳定绿电供应",
            "symbol": "安全绳、风机叶片、海浪、并网指示灯"
        },
    },
    {
        "fixture_id": "enterprise_energy_short_003",
        "title": "一度电的路",
        "industry": "energy_and_utilities",
        "style": "short_video_vertical_world_500_style",
        "duration_seconds": 80,
        "logline": "用竖屏快节奏追踪一度绿电从风机、储能、调度到家庭灯光的路径。",
        "brand_personality": ["清晰", "现代", "可信", "绿色", "高效"],
        "world": {
            "place": "风电场、储能站、调度屏幕、城市家庭",
            "hero": "一位年轻调度员",
            "problem": "普通用户看不见能源系统如何保持稳定",
            "technology": "绿电追踪平台和实时负荷预测",
            "team": "调度员、风机巡检员、储能工程师",
            "outcome": "绿色能源被看见，也被稳定使用",
            "symbol": "一度电图标、流动光线、家庭台灯"
        },
    },
    {
        "fixture_id": "enterprise_auto_business_001",
        "title": "移动的下一公里",
        "industry": "automotive_and_mobility",
        "style": "business_corporate_brand_film",
        "duration_seconds": 155,
        "logline": "从智能制造工厂到城市道路测试，展示汽车集团如何让电动化和智能驾驶进入真实生活。",
        "brand_personality": ["精密", "可靠", "创新", "安全", "全球化"],
        "world": {
            "place": "智能制造工厂、测试道路、城市充电站、家庭车库",
            "hero": "整车安全测试工程师",
            "problem": "智能出行需要在效率、体验和安全之间取得可信平衡",
            "technology": "电驱平台、辅助驾驶安全系统和电池热管理",
            "team": "安全测试工程师、软件工程师、制造技师、用户研究员",
            "outcome": "车辆从工厂测试走向用户真实通勤",
            "symbol": "碰撞假人、测试轨迹、充电光环"
        },
    },
    {
        "fixture_id": "enterprise_auto_story_002",
        "title": "回家的路线",
        "industry": "automotive_and_mobility",
        "style": "story_driven_enterprise_brand_film",
        "duration_seconds": 170,
        "logline": "一位工程师在发布会前夜回到老家，用父亲的一段夜路检验团队对安全的执念。",
        "brand_personality": ["温暖", "安全", "克制", "技术可信", "家庭感"],
        "world": {
            "place": "研发中心、夜间高速、县城老家、试验场",
            "hero": "智能驾驶安全工程师",
            "problem": "真正的安全不是参数，而是家人也愿意信任的夜路",
            "technology": "夜间辅助驾驶、传感器融合和安全冗余系统",
            "team": "安全工程师、测试车队、软件团队、用户体验研究员",
            "outcome": "系统在复杂夜路中保持克制可靠，父子关系完成和解",
            "symbol": "夜路反光线、父亲旧车钥匙、仪表安全提示"
        },
    },
    {
        "fixture_id": "enterprise_auto_short_003",
        "title": "城市三次变道",
        "industry": "automotive_and_mobility",
        "style": "short_video_vertical_world_500_style",
        "duration_seconds": 70,
        "logline": "用三次城市变道串联安全感、智能判断和用户日常体验，形成一支快节奏出行短片。",
        "brand_personality": ["年轻", "敏捷", "安全", "科技感", "不浮夸"],
        "world": {
            "place": "早高峰城市道路、地下车库、充电站、车内座舱",
            "hero": "年轻产品体验官",
            "problem": "城市通勤复杂但用户需要轻松可信的驾驶体验",
            "technology": "城市辅助驾驶、座舱交互、电池补能系统",
            "team": "产品体验官、驾驶安全团队、座舱设计师",
            "outcome": "一次紧张通勤被系统化解成稳定体验",
            "symbol": "转向灯、后视镜提示、充电百分比"
        },
    },
    {
        "fixture_id": "enterprise_finance_business_001",
        "title": "看见风险之前",
        "industry": "financial_services_and_insurance",
        "style": "business_corporate_brand_film",
        "duration_seconds": 150,
        "logline": "展示金融集团如何用风控系统、顾问网络和长期陪伴帮助企业穿越不确定周期。",
        "brand_personality": ["稳健", "理性", "可信赖", "专业", "长期陪伴"],
        "world": {
            "place": "金融风控中心、中小企业工厂、客户会议室、数据监测大厅",
            "hero": "企业金融顾问",
            "problem": "企业经营面对汇率、供应链和现金流多重风险",
            "technology": "实时风控平台和企业信用分析系统",
            "team": "金融顾问、风险分析师、行业研究员、客户经理",
            "outcome": "企业获得更稳健的融资和风险应对方案",
            "symbol": "风险曲线、现金流表、握手但不过度摆拍"
        },
    },
    {
        "fixture_id": "enterprise_finance_documentary_002",
        "title": "长期账户",
        "industry": "financial_services_and_insurance",
        "style": "documentary_enterprise_brand_film",
        "duration_seconds": 185,
        "logline": "跟拍一位养老金融顾问和三组普通家庭，记录长期规划如何改变具体生活选择。",
        "brand_personality": ["温和", "耐心", "可信", "专业", "有边界感"],
        "world": {
            "place": "社区服务中心、家庭餐桌、银行咨询室、养老机构",
            "hero": "养老金融顾问",
            "problem": "普通家庭很难把长期养老需求转化成可执行计划",
            "technology": "长期资产配置工具和风险画像系统",
            "team": "养老顾问、风险评估师、社区服务人员、客户家庭",
            "outcome": "家庭获得清晰规划和可持续的安全感",
            "symbol": "账本、家庭合照、长期规划时间线"
        },
    },
    {
        "fixture_id": "enterprise_finance_story_003",
        "title": "雨天的保单",
        "industry": "financial_services_and_insurance",
        "style": "story_driven_enterprise_brand_film",
        "duration_seconds": 165,
        "logline": "暴雨后，一家小店通过保险理赔和数字服务重新开门，金融服务从抽象承诺变成具体支撑。",
        "brand_personality": ["可靠", "及时", "有人情味", "专业", "不煽情"],
        "world": {
            "place": "暴雨后的街区、小店、理赔服务中心、移动端理赔界面",
            "hero": "小店店主和理赔专员",
            "problem": "突发灾害让小微经营者面临现金流和恢复营业压力",
            "technology": "数字理赔平台和灾害风险评估系统",
            "team": "理赔专员、风控团队、客服人员、社区志愿者",
            "outcome": "小店恢复营业，金融服务的可信度被具体看见",
            "symbol": "雨伞、卷帘门、理赔进度条、重新亮起的招牌"
        },
    },
    {
        "fixture_id": "enterprise_retail_business_001",
        "title": "从货架到生活",
        "industry": "consumer_retail_and_supply_chain",
        "style": "business_corporate_brand_film",
        "duration_seconds": 150,
        "logline": "从全球采购、冷链仓储到门店服务，展示消费零售集团如何把稳定供应变成日常生活体验。",
        "brand_personality": ["高效", "亲近", "可靠", "有规模", "现代"],
        "world": {
            "place": "智慧仓、冷链车、城市门店、家庭厨房",
            "hero": "供应链计划经理",
            "problem": "消费者只看见货架，但背后是复杂供应链的稳定协作",
            "technology": "需求预测系统、冷链追踪和智能补货平台",
            "team": "采购经理、仓储调度、门店员工、冷链司机",
            "outcome": "新鲜商品准时抵达门店和家庭",
            "symbol": "温度标签、补货屏、货架灯"
        },
    },
    {
        "fixture_id": "enterprise_retail_story_002",
        "title": "今晚有饭",
        "industry": "consumer_retail_and_supply_chain",
        "style": "story_driven_enterprise_brand_film",
        "duration_seconds": 160,
        "logline": "一位冷链司机、一名门店员工和一个加班回家的家庭，在同一顿晚饭里被供应链连接起来。",
        "brand_personality": ["温暖", "可靠", "生活化", "克制", "有烟火气"],
        "world": {
            "place": "城市冷链仓、雨夜道路、社区门店、家庭厨房",
            "hero": "冷链司机",
            "problem": "日常生活里的准时与新鲜，需要被看不见的人持续托住",
            "technology": "冷链温控系统和门店补货算法",
            "team": "冷链司机、仓库调度员、门店员工、家庭用户",
            "outcome": "一顿普通晚饭按时发生，品牌价值落在生活里",
            "symbol": "温控箱、雨刷、晚饭蒸汽、门店灯光"
        },
    },
    {
        "fixture_id": "enterprise_retail_short_003",
        "title": "15分钟补货",
        "industry": "consumer_retail_and_supply_chain",
        "style": "short_video_vertical_world_500_style",
        "duration_seconds": 65,
        "logline": "用短视频节奏展示一件爆款商品从缺货预警、仓内拣选、骑手到店到货架补齐的全过程。",
        "brand_personality": ["快", "清楚", "现代", "可靠", "贴近日常"],
        "world": {
            "place": "门店货架、智慧仓、配送路口、收银台",
            "hero": "门店值班经理",
            "problem": "用户需求突然变化时，零售系统需要快速响应但不能失控",
            "technology": "实时库存系统、智能补货和即时配送调度",
            "team": "门店经理、仓内拣选员、配送员、运营调度",
            "outcome": "货架被及时补齐，用户感受到稳定而不是系统复杂度",
            "symbol": "缺货预警、扫码枪、倒计时、货架补齐"
        },
    },
]


def main() -> None:
    collection = {
        "collection_id": "retrieval_review_5_industries_x_3_styles",
        "target_total": 15,
        "existing_fixture": "enterprise_healthtech_story_001",
        "generated_count": len(FIXTURES),
        "fixtures": ["enterprise_healthtech_story_001"] + [fixture["fixture_id"] for fixture in FIXTURES],
        "industries": [
            "medical_technology_and_digital_health",
            "energy_and_utilities",
            "automotive_and_mobility",
            "financial_services_and_insurance",
            "consumer_retail_and_supply_chain",
        ],
        "styles": [
            "business_corporate_brand_film",
            "documentary_enterprise_brand_film",
            "story_driven_enterprise_brand_film",
            "short_video_vertical_world_500_style",
        ],
    }
    write_json(ROOT / "collection_manifest.json", collection)
    for fixture in FIXTURES:
        write_fixture(fixture)


def write_fixture(spec: dict) -> None:
    fixture_dir = ROOT / spec["fixture_id"]
    fixture_dir.mkdir(parents=True, exist_ok=True)
    write_json(fixture_dir / "manifest.json", build_manifest(spec))
    write_json(fixture_dir / "source.json", build_source(spec))
    write_json(fixture_dir / "knowledge.json", build_knowledge(spec))
    write_json(fixture_dir / "retrieval.json", build_retrieval(spec))


def build_manifest(spec: dict) -> dict:
    return {
        "fixture_id": spec["fixture_id"],
        "title": spec["title"],
        "industry": spec["industry"],
        "company_profile": "world_500_style_global_enterprise",
        "film_style": spec["style"],
        "duration_seconds": spec["duration_seconds"],
        "scene_count": 10,
        "layer_files": {
            "source": "source.json",
            "knowledge": "knowledge.json",
            "retrieval": "retrieval.json",
        },
        "prototype_notes": [
            "Fake but plausible enterprise film fixture for retrieval prototype testing.",
            "Does not follow outputs/film_analysis layout.",
            "Embeddings are empty arrays by design.",
        ],
    }


def build_source(spec: dict) -> dict:
    world = spec["world"]
    scenes = []
    for index, (scene_id, role, role_text, _purposes) in enumerate(SCENE_PLAN, start=1):
        scenes.append(
            {
                "scene_id": scene_id,
                "time_range": time_range(index, spec["duration_seconds"]),
                "narrative_role": role,
                "frames": [
                    {
                        "frame_id": f"{scene_id}_f001",
                        "timestamp": frame_timestamp(index, 1),
                        "path": f"frames/{scene_id}/{scene_id}_f001.jpg",
                        "visual_note": f"{world['place']}中的{world['hero']}出现在画面里，承担“{role_text}”的叙事功能。",
                    },
                    {
                        "frame_id": f"{scene_id}_f002",
                        "timestamp": frame_timestamp(index, 2),
                        "path": f"frames/{scene_id}/{scene_id}_f002.jpg",
                        "visual_note": f"画面强调{world['symbol']}，把{world['problem']}转化为可观察的现场细节。",
                    },
                ],
                "package": {
                    "package_id": f"pkg_{scene_id}",
                    "summary": scene_summary(spec, role),
                    "visible_entities": visible_entities(spec, role),
                    "actions": scene_actions(role),
                    "setting": world["place"],
                    "audio_hint": audio_hint(spec["style"], role),
                },
            }
        )
    return {
        "video_id": spec["fixture_id"],
        "title": spec["title"],
        "industry": spec["industry"],
        "style": spec["style"],
        "logline": spec["logline"],
        "source_layer": {
            "frames_root": "frames/",
            "packages_root": "packages/",
            "frame_policy": "fake_keyframes_only",
            "package_policy": "scene_level_package",
        },
        "scenes": scenes,
    }


def build_knowledge(spec: dict) -> dict:
    world = spec["world"]
    return {
        "video_id": spec["fixture_id"],
        "knowledge_layer": {
            "film_level_intent": f"用{spec['style']}表达{spec['industry']}企业如何把{world['technology']}转化为对真实人的支持。",
            "brand_personality": spec["brand_personality"],
            "overall_risk": "如果只强调规模和技术，容易变成汇报片；如果只强调情绪，企业能力会显得不足。",
        },
        "scene_knowledge": [
            {
                "scene_id": scene_id,
                "knowledge_id": f"know_{scene_id}",
                "shooting_techniques": shooting_techniques(spec["style"], role),
                "director_intent": director_intent(spec, role),
                "emotional_function": emotional_function(role),
                "reusable_knowledge": reusable_knowledge(spec, role),
                "best_usage": best_usage(role),
                "risk": risk(role, spec["style"]),
            }
            for scene_id, role, _role_text, _purposes in SCENE_PLAN
        ],
    }


def build_retrieval(spec: dict) -> dict:
    world = spec["world"]
    items = []
    for scene_id, role, role_text, purposes in SCENE_PLAN:
        items.append(
            {
                "retrieval_id": f"ret_{scene_id}",
                "scene_id": scene_id,
                "llm_tags": {
                    "entities": loose_entities(spec, role),
                    "relations": [
                        f"{world['hero']}与{world['team']}的关系",
                        f"{world['technology']}与{world['problem']}的关系",
                        f"世界500强企业能力与真实用户场景",
                    ],
                    "actions_and_expressions": scene_actions(role),
                    "background_and_setting": [world["place"], spec["style"], world["symbol"]],
                    "social_relations": [world["team"], "企业与社会责任", "专业服务对象"],
                },
                "script_use_sentence": script_use_sentence(spec, role, role_text),
                "script_stage": role,
                "creative_purpose": purposes,
                "embedding_texts": {
                    "script_usage": script_usage_embedding(spec, role),
                    "knowledge_semantic": reusable_knowledge(spec, role),
                    "visual_semantic": f"{world['place']}、{world['hero']}、{world['symbol']}、{world['technology']}、{spec['brand_personality'][0]}质感。",
                },
                "embedding_vectors": {
                    "script_usage": [],
                    "knowledge_semantic": [],
                    "visual_semantic": [],
                },
            }
        )
    return {
        "video_id": spec["fixture_id"],
        "retrieval_layer": {
            "tag_policy": "loose_llm_generated_tags_without_synonym_constraint",
            "embedding_policy": "embedding_texts_are_ready_but_vectors_are_empty_until_qwen_embedding_is_enabled",
            "effective_similarity_hint": 0.4,
        },
        "items": items,
    }


def time_range(index: int, duration: int) -> str:
    start = round((index - 1) * duration / 10)
    end = round(index * duration / 10)
    return f"{format_time(start)}-{format_time(end)}"


def format_time(seconds: int) -> str:
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def frame_timestamp(index: int, frame_no: int) -> str:
    seconds = (index - 1) * 15 + frame_no * 5
    return f"{seconds // 60:02d}:{seconds % 60:02d}.{frame_no * 2}"


def scene_summary(spec: dict, role: str) -> str:
    world = spec["world"]
    mapping = {
        "opening": f"用{world['place']}建立真实业务场景和{world['problem']}。",
        "setup": f"把{world['problem']}具体化，让企业能力有进入理由。",
        "character_intro": f"通过{world['hero']}进入专业视角，避免空泛品牌叙述。",
        "team_work": f"展现{world['team']}如何围绕同一个问题协作。",
        "technology_showcase": f"让{world['technology']}作为解决路径自然出现。",
        "scale_reveal": f"揭示世界500强企业的网络、系统和跨地域能力。",
        "value_expression": f"用具体沟通和责任动作表达{spec['brand_personality'][0]}与可信赖。",
        "outcome": f"呈现{world['outcome']}，但不过度承诺。",
        "growth": f"把长期主义和改进动作落在具体人物身上。",
        "ending": f"回到核心场景，完成{spec['title']}的品牌收束。",
    }
    return mapping[role]


def visible_entities(spec: dict, role: str) -> list[str]:
    world = spec["world"]
    base = [world["hero"], world["technology"], world["symbol"]]
    by_role = {
        "opening": [world["place"], "真实用户场景", "环境细节"],
        "setup": ["一线人员", "等待中的客户或用户", "问题线索"],
        "character_intro": [world["hero"], "工作证件", "专业工具"],
        "team_work": world["team"].split("、"),
        "technology_showcase": [world["technology"], "操作界面", "确认提示"],
        "scale_reveal": ["全球地图", "多地团队", "数据看板"],
        "value_expression": ["面对面沟通", "记录动作", "确认反馈"],
        "outcome": ["服务对象", "结果记录", "稳定表情"],
        "growth": ["年轻员工", "反馈信息", "版本记录"],
        "ending": ["品牌字幕", world["symbol"], "回到开场地点"],
    }
    return dedupe(base + by_role[role])[:8]


def scene_actions(role: str) -> list[str]:
    mapping = {
        "opening": ["进入现场", "观察环境", "问题显现"],
        "setup": ["比对信息", "等待确认", "压力累积"],
        "character_intro": ["整理工具", "看向现场", "做出判断"],
        "team_work": ["围绕问题讨论", "标注风险点", "分配下一步"],
        "technology_showcase": ["打开系统", "验证数据", "确认流程"],
        "scale_reveal": ["多地连线", "同步信息", "网络展开"],
        "value_expression": ["面对面解释", "认真记录", "确认承诺"],
        "outcome": ["方案落地", "用户点头", "情绪稳定"],
        "growth": ["接收反馈", "复盘问题", "继续改进"],
        "ending": ["回到起点", "主题字幕出现", "音乐收束"],
    }
    return mapping[role]


def audio_hint(style: str, role: str) -> str:
    if "short_video" in style:
        return "快节奏切点、清晰提示音、字幕节奏跟随动作。"
    if "documentary" in style:
        return "保留同期声、人声和环境声，音乐克制后置。"
    if "story_driven" in style:
        return "音乐跟随人物情绪推进，关键处降低配乐突出真实声音。"
    if role in {"opening", "ending"}:
        return "稳重企业音乐，低频铺底，收束时保留环境声。"
    return "稳定节奏的商务配乐，突出专业感和清晰叙事。"


def shooting_techniques(style: str, role: str) -> list[str]:
    style_base = {
        "business_corporate_brand_film": ["稳定轨道运动", "干净构图", "克制数据可视化"],
        "documentary_enterprise_brand_film": ["手持跟拍", "同期声保留", "自然光观察"],
        "story_driven_enterprise_brand_film": ["人物视角推进", "首尾呼应", "情绪留白"],
        "short_video_vertical_world_500_style": ["竖屏构图", "快速信息切片", "动作驱动转场"],
    }
    role_extra = {
        "opening": ["远景建立场域"],
        "setup": ["中近景建立压力"],
        "character_intro": ["人物工作细节特写"],
        "team_work": ["群像调度"],
        "technology_showcase": ["界面与人同框"],
        "scale_reveal": ["多地并置剪辑"],
        "value_expression": ["正反打沟通"],
        "outcome": ["结果细节特写"],
        "growth": ["夜景或复盘镜头"],
        "ending": ["回到开场空间"],
    }
    return style_base.get(style, ["稳定构图"]) + role_extra[role]


def director_intent(spec: dict, role: str) -> str:
    world = spec["world"]
    if role == "technology_showcase":
        return f"把{world['technology']}拍成服务人的工具，而不是炫技符号。"
    if role == "team_work":
        return f"证明能力来自{world['team']}的协作，而不是单个英雄人物。"
    if role == "value_expression":
        return f"用具体互动表达{spec['brand_personality'][0]}和可信赖，避免口号。"
    return scene_summary(spec, role)


def emotional_function(role: str) -> str:
    mapping = {
        "opening": "建立真实感和进入感。",
        "setup": "制造必要的张力和问题意识。",
        "character_intro": "让观众找到可信的人物入口。",
        "team_work": "形成组织能力和协作能量。",
        "technology_showcase": "把不确定转化为可执行路径。",
        "scale_reveal": "扩大格局但保持克制。",
        "value_expression": "建立信任和价值认同。",
        "outcome": "释放压力并确认价值落地。",
        "growth": "形成长期主义和内部驱动力。",
        "ending": "完成主题闭环和品牌余韵。",
    }
    return mapping[role]


def reusable_knowledge(spec: dict, role: str) -> str:
    world = spec["world"]
    mapping = {
        "opening": f"{spec['industry']}企业片可以先拍真实问题，再进入企业能力。",
        "setup": "需求建立必须来自具体人物和场景，不应直接从产品卖点开始。",
        "character_intro": "人物介绍要通过专业动作建立可信，而不是靠头衔字幕。",
        "team_work": "团队协作要有明确讨论对象，避免空泛办公蒙太奇。",
        "technology_showcase": f"表现{world['technology']}时要让技术和人的判断同框。",
        "scale_reveal": "世界500强规模感适合通过网络、流程和服务覆盖表达。",
        "value_expression": "价值观要通过面对面沟通、记录、确认和承担责任体现。",
        "outcome": "结果段落要落地但克制，避免夸大承诺。",
        "growth": "成长段落最好由真实反馈推动，而不是靠奋斗口号推动。",
        "ending": "结尾回到开场问题，会让品牌承诺显得像已经发生的行动。",
    }
    return mapping[role]


def best_usage(role: str) -> str:
    return {
        "opening": "适合开场、问题建立、真实场景进入。",
        "setup": "适合需求铺垫、矛盾建立、一线压力表达。",
        "character_intro": "适合人物出场、专业身份建立。",
        "team_work": "适合团队协作、组织能力、跨职能工作流。",
        "technology_showcase": "适合技术能力、人机协作、系统价值展示。",
        "scale_reveal": "适合规模揭示、全球网络、企业能力展开。",
        "value_expression": "适合建立信任、表达责任、价值观落地。",
        "outcome": "适合结果落地、情绪释放、客户价值确认。",
        "growth": "适合员工成长、长期主义、使命升温。",
        "ending": "适合片尾收束、品牌价值表达、温和号召。",
    }[role]


def risk(role: str, style: str) -> str:
    base = {
        "opening": "如果后续没有人物承接，容易成为空镜。",
        "setup": "如果压力过度，会显得企业在制造焦虑。",
        "character_intro": "如果头衔感太重，会像领导介绍。",
        "team_work": "如果只拍会议和笑脸，会变成模板化办公片。",
        "technology_showcase": "如果界面过多，会像功能说明视频。",
        "scale_reveal": "如果地图和数据过多，会变成汇报片。",
        "value_expression": "如果靠字幕解释价值观，会显得说教。",
        "outcome": "如果承诺过满，会引发可信度风险。",
        "growth": "如果拍成加班热血，会显得陈旧。",
        "ending": "如果品牌字幕太满，会破坏克制质感。",
    }[role]
    if "short_video" in style:
        return base + " 竖屏节奏还要避免信息过载。"
    return base


def loose_entities(spec: dict, role: str) -> list[str]:
    return visible_entities(spec, role) + [spec["industry"], spec["brand_personality"][0]]


def script_use_sentence(spec: dict, role: str, role_text: str) -> str:
    world = spec["world"]
    return f"适合用于{role}段落，{role_text}，用{world['hero']}、{world['technology']}和{world['symbol']}表达{spec['brand_personality'][0]}的世界500强品牌质感。"


def script_usage_embedding(spec: dict, role: str) -> str:
    world = spec["world"]
    return f"{best_usage(role)} {spec['title']} {spec['industry']} {spec['style']} {world['problem']} {world['technology']} {world['outcome']}"


def dedupe(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
