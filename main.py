import os

import torch

from moudel.TransformerChat import ChatbotTransformer
from qadata.QaDataDict import vocab, data_x, data_y
from response.ResponseGenerator import ResponseGenerator
from train.TrainModel import TrainModel

model_path = "data/model.pt"
if os.path.exists(model_path):  # 检查文件是否存在
    model = ChatbotTransformer(input_dim=16, output_dim=len(vocab), nhead=8, num_encoder_layers=1, num_decoder_layers=1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # 选择合适的设备
    model.eval()  # 将模型设置为评估模式，关闭dropout等

else:
    print("Model not found. Training a new model.")
    model = ChatbotTransformer(input_dim=16, output_dim=len(vocab), nhead=8, num_encoder_layers=1, num_decoder_layers=1)
    # 训练器实例化
    trainer = TrainModel(model)
    trainer.train(data_x, data_y, 30)

# 响应生成器实例化
generator = ResponseGenerator(model)
response = generator.generate_response("张千")
print("\nGenerated response:", response)
