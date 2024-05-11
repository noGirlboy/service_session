import torch
import logging
from qadata.QaDataDict import word_to_idx, idx_to_word, to_idx_seq

# 设置日志记录
logging.basicConfig(level=logging.INFO)


class ResponseGenerator:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device).eval()
        # 确保word_to_idx和idx_to_word与训练时一致
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.device = device

    def generate_response(self, input_sentence, max_length=50):
        # 输入验证和清理
        if not isinstance(input_sentence, str) or not input_sentence.strip():
            logging.error("输入句子无效。")
            return []

        input_seq = torch.tensor(to_idx_seq(input_sentence)).to(self.device)
        target_seq = torch.tensor([self.word_to_idx["<SOS>"]]).to(self.device)  # 使用特殊的起始标记

        # 动态序列生成
        generated_sequence = self._generate_sequence(input_seq, target_seq, max_length)

        return generated_sequence[1:-1]  # 跳过起始和结束标记

    def _generate_sequence(self, input_seq, target_seq, max_length):
        with torch.no_grad():
            generated_sequence = []
            for i in range(max_length):
                output = self.model(input_seq.unsqueeze(0), target_seq.unsqueeze(0))
                output_token = output.argmax(2)[-1].item()

                if output_token == self.word_to_idx["<EOS>"]:
                    # 如果遇到结束标记，则提前终止生成
                    break

                generated_sequence.append(idx_to_word[output_token])
                target_seq = torch.cat((target_seq, torch.tensor([output_token]).to(self.device)), dim=0)

            logging.info("生成的句子: {}".format(" ".join(generated_sequence)))
            return generated_sequence
