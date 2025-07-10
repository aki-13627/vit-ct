import torch.nn as nn
from transformers import ViTForImageClassification

def create_vit_model(num_classes= 2, pretrained=True):
    model_name = 'google/vit-base-patch16-224'
    if pretrained:
        model = ViTForImageClassification.from_pretrained(model_name)
    else:
        config = ViTForImageClassification.from_pretrained(model_name).config
        model = ViTForImageClassification(config)
        
    num_original_features = model.classifier.in_features
    
    model.classifier = nn.Linear(
        in_features=num_original_features,
        out_features=num_classes
    )        
    
    return model

if __name__ == '__main__':
    print(f'---テストモデルの作成')
    test_model = create_vit_model(num_classes= 2)
    print(f'---モデルの構造---')
    # print(test_model) #全体を表示すると長すぎるので、分類層だけを表示
    
    print("\nカスタマイズされた分類層:")
    print(test_model.classifier)
    
    print("\n---テスト終了---")
