from gliner import GLiNER

def extract_basic():
    # 加载模型 (首次运行会自动下载)
    # "urchade/gliner_medium-v2.1" 是目前平衡性很好的版本
    print("正在加载 GLiNER 模型...")
    model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

    text = "马斯克昨天在旧金山宣布，SpaceX 将在下个月发射新的火箭。"
    
    # 定义我们要提取的标签
    labels = ["person", "location", "organization"]
    
    print(f"\n待处理文本: {text}")
    print(f"目标标签: {labels}")

    # 执行预测
    entities = model.predict_entities(text, labels)

    print("\n--- 抽取结果 ---")
    for entity in entities:
        print(f"{entity['text']} => {entity['label']} (置信度: {entity['score']:.2f})")

if __name__ == "__main__":
    extract_basic()
