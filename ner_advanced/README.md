# Advanced NER Project (GLiNER)

本项目演示使用 **GLiNER** (Generalist Model for Named Entity Recognition) 进行实体抽取。

GLiNER 是一个轻量级、高性能的 NER 模型，它的核心优势在于：
1.  **Zero-shot**: 可以在推理时动态指定任意实体标签，无需重新训练。
2.  **Local**: 完全本地运行，无需 API Key，数据更安全。
3.  **Fast**: 速度远快于大型生成式 LLM。

## 环境准备

1.  **安装依赖**:
    ```bash
    uv sync
    ```
    *(注意：首次运行代码时会自动下载模型权重，约 600MB)*

## 运行示例

### 1. 基础用法
演示加载模型并提取标准实体（人名、地点、组织）。
```bash
uv run 1_gliner_basic.py
```

### 2. 自定义标签 (Zero-shot)
演示 GLiNER 的强大之处：提取任意自定义标签（如“价格”、“日期”、“产品特性”等）。
```bash
uv run 2_gliner_custom.py
```
