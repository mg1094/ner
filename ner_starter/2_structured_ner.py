import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from typing import List

# 加载环境变量
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# 定义我们期望的数据结构 (虽然 JSON Mode 不直接使用 Pydantic 类，但定义它有助于我们写 Prompt)
class EntityResult(BaseModel):
    person: List[str]
    location: List[str]
    organization: List[str]

def extract_entities_structured(text):
    """
    使用 JSON Mode 强制模型输出结构化数据
    """
    system_prompt = """
你是一个实体抽取助手。请从用户提供的文本中提取实体。
必须严格输出合法的 JSON 格式。
JSON 结构应包含: "person" (list), "location" (list), "organization" (list)。
"""

    user_prompt = f"分析文本: {text}"

    print(f"--- 发送请求 (JSON Mode) ---")

    response = client.chat.completions.create(
        model="pa/gemini-3-pro-preview", # 确保模型支持 json_object 模式 (gpt-4-1106-preview, gpt-3.5-turbo-1106 等)
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}, # 关键参数：强制 JSON 输出
        temperature=0
    )

    content = response.choices[0].message.content
    print("content:",content)
    
    # 解析 JSON
    try:
        data = json.loads(content)
        return data
    except json.JSONDecodeError:
        print("Error: 模型输出的不是有效的 JSON")
        return None

if __name__ == "__main__":
    text = "苹果公司 CEO 库克在加利福尼亚州的发布会上推出了新款 iPhone。"
    print(f"待处理文本: {text}\n")
    
    result = extract_entities_structured(text)
    
    print(f"--- 模型输出 (Parsed Dict) ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 验证一下是否可以像字典一样访问
    if result:
        print(f"\n提取到的人名: {result.get('person')}")
