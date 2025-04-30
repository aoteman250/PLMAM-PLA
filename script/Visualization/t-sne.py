import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import sys
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dataset import *
from model_tsne import MultiViewNet


print(sys.argv)

SHOW_PROCESS_BAR = True
data_path = '../data/'
device = torch.device("cuda")

batch_size = 5

model = MultiViewNet()

model = model.to(device)

data_loaders = {phase_name:
                    DataLoader(MyDataset(data_path, phase_name,
                                         max_smi_len=120, max_seq_len=1000),
                               batch_size=batch_size,
                               shuffle=True,
                               )

                for phase_name in ['training', 'validation', 'test',]}

loss_function = nn.MSELoss(reduction='sum')  #

start = datetime.now()
print('start at ', start)

def test(model: nn.Module, test_loader, device, show, _p):
    model.eval()
    all_intermediate_outputs = []
    all_y = []

    with torch.no_grad():
        for idx, (*x, y) in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):
            smile = x[0].to(device)
            sequence = x[1].to(device)
            smi = x[2].to(device)
            seq = x[3].to(device)
            y = y.to(device)

            output, intermediate_outputs = model(smile, sequence, smi, seq)
            all_intermediate_outputs.append(intermediate_outputs)
            all_y.append(y)
            all_y1 = torch.cat(all_y)

    return all_y1, torch.cat(all_intermediate_outputs)

model.load_state_dict(torch.load('trained_model.pt'))

all_y = []
all_logits1 = []
all_logits = []
all_compoundFeature_onehot1 = []
all_proteinFeature_onehot1 = []

for _p in ['test']:
    y,  intermediate_outputs = test(model, data_loaders[_p], device, SHOW_PROCESS_BAR, _p)
    all_y.append(y)
    all_logits1.append(intermediate_outputs)


# 将所有批次的中间输出拼接在一起
all_logits1 = torch.cat(all_logits1)
all_y = torch.cat(all_y)


# 应用t-SNE降维
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000)  # 增大perplexity适应大数据
fused_features_2d = tsne.fit_transform(all_logits1.cpu().numpy())

## 归一化真实值y
min_value = all_y.min().item()
max_value = all_y.max().item()
normalized_y = (all_y - min_value) / (max_value - min_value)

# 创建一个颜色映射，例如，使用归一化后的y的真实值来决定颜色
colors = plt.cm.coolwarm(normalized_y.cpu().numpy())

# 绘制t-SNE降维后的数据点
plt.figure(figsize=(10, 8))
scatter = plt.scatter(fused_features_2d[:, 0], fused_features_2d[:, 1], c=colors, alpha=0.8)
# 设定坐标轴范围

sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=min(normalized_y), vmax=max(normalized_y)))
plt.colorbar(sm)
plt.savefig('tsne.eps',dpi=300, format='eps',)
plt.show()

