# Earth-Agent：地球科学 AI 智能体

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Stars](https://img.shields.io/github/stars/opendatalab/Earth-Agent)](https://github.com/opendatalab/Earth-Agent/stargazers)

首个面向地球观测数据的多模态 AI 智能体系统，由**上海人工智能实验室**与**中山大学**联合开发。目前支持国内主要大模型后端。

## 核心能力

Earth-Agent 集成了 **约 130 个专业遥感分析工具**，覆盖五大领域：

| 类别 | 工具数 | 说明 |
|------|--------|------|
| 遥感指数 (Index) | 22 | NDVI, EVI, NDWI, NDBI 等常用指数计算 |
| 参数反演 (Inversion) | 21 | 地表温度、反照率、发射率等定量反演 |
| 图像感知 (Perception) | 16 | 阈值分割、目标检测辅助、图像处理 |
| 统计分析 (Statistics) | 61 | 基本统计、PCA、时间序列分析 |
| 趋势分析 (Analysis) | 10 | Mann-Kendall、Sen's Slope、变点检测 |

## 架构

```
用户指令 → LangChain Agent (ReAct 模式)
              ↓
         MCP 工具调用
              ↓
    ┌─────────────────────────┐
    │ Index | Inversion | Per │
    │ ception | Stats | Anal  │
    └─────────────────────────┘
```

## 支持的 LLM 后端

- DeepSeek-V3/R1
- GPT-4o / GPT-5
- Claude 3.7 / 4
- Qwen3 / Qwen3-Max
- Kimi K2
- GLM-4.5
- InternLM-3
- Llama 4
- Gemini 2.5
- 更多见 `agent/config_*.json`

## 快速开始

```bash
git clone https://github.com/opendatalab/Earth-Agent.git
cd Earth-Agent
pip install -r requirements.txt

# 配置 LLM
cp agent/config_gpt5.json.example agent/config.json
# 编辑 config.json 填入 API key

# 运行
python langchain_gpt_enhanced.py
```

## 项目结构

```
Earth-Agent/
├── agent/
│   ├── tools/        # 130 个遥感工具
│   │   ├── Index.py
│   │   ├── Inversion.py
│   │   ├── Perception.py
│   │   ├── Analysis.py
│   │   └── Statistics.py
│   ├── config_*.json # LLM 后端配置
│   └── main_agent.py # 核心 Agent 逻辑
├── benchmark/        # 评测基准
├── docs/             # 文档
│   └── tools-reference.md  # 工具参考手册
└── README.md
```

## 工具参考

详细工具列表与使用示例见 [docs/tools-reference.md](docs/tools-reference.md)。

## 引用

如使用 Earth-Agent，请引用原论文。

## 贡献

欢迎提交 Issue 和 PR。本项目接受功能扩展、Bug 修复、文档完善等多种形式的贡献。
