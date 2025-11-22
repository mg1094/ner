import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") # 兼容其他 OpenAI 格式的 API
)

def extract_entities_simple(text):
    """
    使用基础 Prompt 进行实体抽取 (Zero-shot / Few-shot)
    """
    prompt = f"""
你是一个专业的实体抽取助手。请从以下文本中提取出【人名 (Person)】、【地点 (Location)】和【组织机构 (Organization)】。

文本: "{text}"

请按以下格式输出:
人名: [人名1, 人名2]
地点: [地点1, 地点2]
组织机构: [组织1, 组织2]

如果没有找到某类实体，请输出空列表 []。
"""
    
    print(f"--- 请求 Prompt ---\n{prompt.strip()}\n------------------")

    response = client.chat.completions.create(
        model="pa/gemini-3-pro-preview", # 或者 gpt-3.5-turbo, deepseek-chat 等
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0 # 设置为 0 以获得更稳定的输出
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    text = "马斯克昨天在旧金山宣布，SpaceX 将在下个月发射新的火箭。"
    print(f"待处理文本: {text}\n")
    
    result = extract_entities_simple(text)
    print(f"--- 模型输出 ---\n{result}")
