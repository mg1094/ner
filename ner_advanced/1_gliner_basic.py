from gliner import GLiNER

def extract_basic():
    # 加载模型 (首次运行会自动下载)
    # "urchade/gliner_medium-v2.1" 是目前平衡性很好的版本
    print("正在加载 GLiNER 模型...")
    model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    
    # 打印模型参数信息
    print("\n=== 模型参数信息 ===")
    if hasattr(model, 'model') and hasattr(model.model, 'config'):
        config = model.model.config
        print(f"模型名称: {config.name_or_path if hasattr(config, 'name_or_path') else 'N/A'}")
        print(f"模型类型: {config.model_type if hasattr(config, 'model_type') else 'N/A'}")
        
        # 最大长度相关参数
        if hasattr(config, 'max_position_embeddings'):
            print(f"最大位置嵌入: {config.max_position_embeddings}")
        if hasattr(config, 'max_length'):
            print(f"最大长度: {config.max_length}")
        
        # 模型维度
        if hasattr(config, 'hidden_size'):
            print(f"隐藏层维度: {config.hidden_size}")
        if hasattr(config, 'num_attention_heads'):
            print(f"注意力头数: {config.num_attention_heads}")
        if hasattr(config, 'num_hidden_layers'):
            print(f"隐藏层数: {config.num_hidden_layers}")
    
    # Tokenizer 信息
    if hasattr(model, 'model') and hasattr(model.model, 'tokenizer'):
        tokenizer = model.model.tokenizer
        print(f"\nTokenizer 信息:")
        if hasattr(tokenizer, 'model_max_length'):
            print(f"  Tokenizer 最大长度: {tokenizer.model_max_length}")
        if hasattr(tokenizer, 'vocab_size'):
            print(f"  词汇表大小: {tokenizer.vocab_size}")
        print(f"  Tokenizer 类型: {type(tokenizer).__name__}")
    
    # 参数量统计
    try:
        import torch
        if hasattr(model, 'model'):
            total_params = sum(p.numel() for p in model.model.parameters())
            trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            print(f"\n参数量统计:")
            print(f"  总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
            print(f"  可训练参数: {trainable_params:,}")
    except:
        pass
    
    print("=" * 30)

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
