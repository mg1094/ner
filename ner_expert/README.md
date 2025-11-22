# Expert NER Project (LoRA Fine-tuning)

本项目演示如何使用 **LoRA (Low-Rank Adaptation)** 技术对大模型进行微调，使其成为特定领域的 NER 专家。

## 场景假设
我们需要从魔法世界的古籍中提取“药剂成分”和“药剂功效”。通用模型可能不知道“龙鳞草”是啥，或者不知道如何按特定格式提取。我们将通过微调教会它。

## 环境准备

1.  **安装依赖**:
    ```bash
    uv sync
    ```

2.  **硬件要求**:
    - 推荐 NVIDIA GPU (8GB+ VRAM)。
    - Mac 用户可以使用 MPS (Metal) 加速，但速度较慢。
    - 本演示使用极小的模型 `Qwen/Qwen2.5-0.5B-Instruct`，确保大多数机器都能跑通。

## 运行流程

### 1. 开始微调
运行训练脚本。这会加载基础模型，读取 `data/sample_dataset.json`，并开始训练。
```bash
uv run train_lora.py
```
*训练完成后，LoRA 权重将保存在 `lora_output` 目录。*

### 2. 模型推理
加载微调后的权重进行测试。
```bash
uv run inference.py
```
