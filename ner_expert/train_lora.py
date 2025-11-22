import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer

# 1. 配置参数
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" # 使用极小模型以便演示
OUTPUT_DIR = "lora_output"
DATA_PATH = "data/sample_dataset.json"

def train():
    print(f"正在加载模型: {MODEL_NAME}...")
    
    # 2. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token # 设置 pad_token

    # 3. 加载模型
    # device_map="auto" 会自动检测 GPU/MPS/CPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # 4. 配置 LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8,            # LoRA 秩，越大参数越多
        lora_alpha=32,  # LoRA 缩放系数
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"] # 通常微调 Attention 层的投影矩阵
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters() # 打印可训练参数量

    # 5. 加载数据
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    # 6. 格式化数据函数 (适配 Qwen/Llama 的 Chat 格式)
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['input'])):
            text = f"Instruction: {example['instruction'][i]}\nInput: {example['input'][i]}\nOutput: {example['output'][i]}"
            output_texts.append(text)
        return output_texts

    # 7. 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        max_steps=10, # 演示仅跑 10 步
        save_steps=10,
        fp16=torch.cuda.is_available(), # GPU 上开启混合精度
        use_mps_device=torch.backends.mps.is_available(), # Mac 上开启 MPS
    )

    # 8. 初始化 Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )

    print("开始训练...")
    trainer.train()
    
    print(f"训练完成！模型已保存至 {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    train()
