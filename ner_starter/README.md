# LLM NER Starter Project

这是一个用于学习使用大模型进行实体抽取 (NER) 的入门项目。

## 环境准备

1.  **安装 uv**:
    如果您尚未安装 `uv`，请运行：
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **安装依赖**:
    ```bash
    uv sync
    ```

3.  **配置 API Key**:
    复制 `.env.example` 为 `.env`，并填入您的 OpenAI API Key (或兼容的 API Key)。
    ```bash
    cp .env.example .env
    ```

## 运行示例

### 1. 基础 Prompt 抽取 (Zero-shot / Few-shot)
演示如何通过自然语言提示让模型提取实体。
```bash
uv run 1_simple_ner.py
```

### 2. 结构化抽取 (JSON Mode)
演示如何强制模型输出标准的 JSON 格式，这是生产环境中最推荐的方式。
```bash
uv run 2_structured_ner.py
```

## 进阶学习
查看项目根目录下的 `llm_ner_report.md` 获取更多进阶策略（如 GLiNER, Fine-tuning）。
