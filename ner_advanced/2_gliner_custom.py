from gliner import GLiNER

def extract_custom():
    print("正在加载 GLiNER 模型...")
    model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

    # 一段包含复杂信息的文本
    text = "新款 iPhone 15 Pro Max 售价 9999 元起，将于 9 月 22 日正式发售，搭载 A17 Pro 芯片。"
    
    # 我们可以随意定义标签，模型会尝试理解语义
    # 这里的标签在训练集中可能从未出现过
    labels = ["product", "price", "date", "feature"]
    
    print(f"\n待处理文本: {text}")
    print(f"自定义标签: {labels}")

    entities = model.predict_entities(text, labels)

    print("\n--- 抽取结果 (Zero-shot) ---")
    for entity in entities:
        print(f"{entity['text']} => {entity['label']} (置信度: {entity['score']:.2f})")

if __name__ == "__main__":
    extract_custom()
