import random
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from datetime import datetime
from qadata.QaDataDict import vocab

# 设定随机种子以确保实验可复现性
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

BATCH_SIZE = 32
PATIENCE = 30  # 早停的耐心值
TRAIN_LOSS = []
VALIDATION_LOSS = []
GRAD_CLIP = 5.0
criterion = nn.CrossEntropyLoss()


class TrainModel:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device

    def define_optimizer(self):
        """定义优化器"""
        # return optim.Adam(self.model.parameters(), lr=0.01)
        return optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)

    def train_epoch(self, data_x, data_y, optimizer):
        """训练单个epoch"""
        total_loss = 0
        for i in range(len(data_x)):
            optimizer.zero_grad()

            input_seq = torch.tensor(data_x[i]).to(self.device)
            target_seq = torch.tensor(data_y[i]).to(self.device)

            for j in range(len(target_seq) - 1):
                if random.random() < 0.5:
                    j = random.randint(0, len(target_seq) - 2)
                output = self.model(input_seq.unsqueeze(0), target_seq[:j + 1].unsqueeze(0))
                loss = criterion(output.view(-1, len(vocab)), target_seq[1:j + 2].view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)  # 梯度裁剪
                optimizer.step()
                total_loss += loss.item()

        return total_loss / len(data_x)

    def train(self, data_x, data_y, epochs):
        optimizer = self.define_optimizer()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(data_x, data_y, optimizer)
            TRAIN_LOSS.append(train_loss)

            # 假设有一个validate函数，返回验证集上的损失
            val_loss = self.validate(data_x, data_y)  # 需要实现validate函数
            VALIDATION_LOSS.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # # 保存模型
                # model_file_name = f'data/model_{datetime.now().strftime("%Y%m%d-%H%M%S")}.pt'
                # torch.save(self.model.state_dict(), model_file_name)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print("早停机制触发，训练终止。")
                    break

            if (epoch + 1) % 10 == 0:
                print('Epoch [{}/{}], Train Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch + 1, epochs, train_loss,
                                                                                          val_loss))

        # 结束训练就绘图
        print("Training completed.")
        self.plot_loss()
        # 保存模型
        model_file_name = f'data/model_{datetime.now().strftime("%Y%m%d-%H%M%S")}.pt'
        torch.save(self.model.state_dict(), model_file_name)

    def validate(self, val_data_x, val_data_y):
        total_val_loss = 0
        for i in range(len(val_data_x)):
            input_seq = torch.tensor(val_data_x[i]).to(self.device)
            target_seq = torch.tensor(val_data_y[i]).to(self.device)

            # 对于验证集，不需要梯度计算和反向传播
            with torch.no_grad():
                for j in range(len(target_seq) - 1):
                    if random.random() < 0.5:
                        j = random.randint(0, len(target_seq) - 2)
                    output = self.model(input_seq.unsqueeze(0), target_seq[:j + 1].unsqueeze(0))
                    loss = criterion(output.view(-1, len(vocab)), target_seq[1:j + 2].view(-1))
                    total_val_loss += loss.item()

        return total_val_loss / len(val_data_x)

    def plot_loss(self):
        """绘制损失曲线"""
        epochs_range = range(1, len(TRAIN_LOSS) + 1)
        plt.plot(epochs_range, TRAIN_LOSS, label='Training Loss')
        plt.plot(epochs_range, VALIDATION_LOSS, label='Validation Loss')
        plt.legend(loc='lower right')
        plt.title('LOSS')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.show()
