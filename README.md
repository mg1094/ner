# LLM NER (Named Entity Recognition) 项目

这是一个完整的 LLM 实体抽取学习项目，包含从入门到专家级别的三个层次实现。

## 项目结构

- **ner_starter/**: 入门项目 - 演示基础的 Prompt Engineering 和结构化输出
- **ner_advanced/**: 进阶项目 - 使用 GLiNER 进行 Zero-shot 实体抽取
- **ner_expert/**: 专家项目 - 使用 LoRA 微调大模型进行领域特定实体抽取

## 快速开始

每个子项目都有独立的 README 文档，请查看对应目录下的 README.md 了解详细使用方法。

### 入门项目
```bash
cd ner_starter
uv sync
uv run 1_simple_ner.py
```

### 进阶项目
```bash
cd ner_advanced
uv sync
uv run 1_gliner_basic.py
```

### 专家项目
```bash
cd ner_expert
uv sync
uv run train_lora.py
```

## 详细文档

查看 `llm_ner_report.md` 了解完整的 NER 最佳实践和前沿方案。

## 技术栈

- Python 3.x
- uv (Python 包管理器)
- OpenAI API / 开源 LLM
- GLiNER
- LoRA / QLoRA 微调技术

