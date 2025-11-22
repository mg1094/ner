import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = "lora_output"

def inference():
    print(f"正在加载基础模型: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    print(f"正在加载 LoRA 权重: {LORA_PATH}...")
    # 加载 LoRA 适配器
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    # 测试输入
    instruction = "从文本中提取魔法药剂的成分 (Ingredient) 和功效 (Effect)。"
    input_text = "将三滴人鱼的眼泪滴入沸水中，饮用后可以在水下呼吸一小时。"
    
    prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(f"\n输入: {input_text}")
    print("正在生成...")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            temperature=0.1
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 简单的后处理，只显示 Output 之后的内容
    if "Output:" in result:
        final_output = result.split("Output:")[-1].strip()
    else:
        final_output = result

    print(f"\n--- 预测结果 ---\n{final_output}")

if __name__ == "__main__":
    inference()
