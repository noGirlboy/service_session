import torch
from torch import nn
from qadata.QaDataDict import vocab


class ChatbotTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, num_encoder_layers, num_decoder_layers):
        super(ChatbotTransformer, self).__init__()
        # 参数校验
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive integers.")
        if nhead <= 0 or num_encoder_layers <= 0 or num_decoder_layers <= 0:
            raise ValueError("nhead, num_encoder_layers, and num_decoder_layers must be positive integers.")
        if len(vocab) <= 0:
            raise ValueError("vocab size must be a positive integer.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(len(vocab), input_dim)
        self.transformer = nn.Transformer(d_model=input_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, src, tgt):
        # 输入校验
        if not isinstance(src, torch.Tensor) or not isinstance(tgt, torch.Tensor):
            raise TypeError("src and tgt must be torch.Tensor")
        if src.dim() != 2 or tgt.dim() != 2:
            raise ValueError("src and tgt must be 2D tensors")

        src = self.embedding(src).permute(1, 0, 2)  # 调整输入张量维度
        tgt = self.embedding(tgt).permute(1, 0, 2)  # 调整输入张量维度
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output
