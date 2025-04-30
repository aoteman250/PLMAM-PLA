import sys
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dataset import *
from model_re import MultiViewNet
import os
import metrics
import torch
import pandas as pd
import csv
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

print(sys.argv)

SHOW_PROCESS_BAR = True
data_path = '../data/'
device = torch.device("cuda")

batch_size = 1

model = MultiViewNet()

model = model.to(device)

data_loaders = {phase_name:
                    DataLoader(MyDataset(data_path, phase_name,
                                         max_smi_len=120, max_seq_len=1000),
                               batch_size=batch_size,
                               shuffle=False,
                               )

                for phase_name in ['re']}


loss_function = nn.MSELoss(reduction='sum')  #
start = datetime.now()
print('start at ', start)

def test(model: nn.Module, test_loader, device, show, _p):
    model.eval()
    outputs = []

    with torch.no_grad():
        for idx, (*x, y) in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):
            smile = x[0].to(device)
            sequence = x[1].to(device)
            smi = x[2].to(device)
            seq = x[3].to(device)

            y_hat,pro,pro1 = model(smile, sequence, smi, seq)
            pro =  pro.mean(dim=2)
            pro1=  pro1.mean(dim=2)
            attn_merged = (pro + pro1) / 2


    return y_hat,pro,pro1,attn_merged


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib


def plot_1d_attention_heatmap(attention, sequence, title='', save_path='attention_heatmap.eps'):
    """
    显示第 121~143 个残基的注意力热图（索引为 120~142），热图为正方格，每 5 个显示编号。
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib import cm

    # ✅ 提取残基范围：索引 120~143（含前不含后），即编号 121~143
    attention = np.asarray(attention[120:143])
    sequence = sequence[120:143]

    # 归一化处理
    min_val, max_val = attention.min(), attention.max()
    if max_val > min_val:
        attention = (attention - min_val) / (max_val - min_val)
    else:
        attention = np.zeros_like(attention)

    attention_2d = attention[np.newaxis, :]  # 转为 2D 矩阵 [1, 23]

    fig, ax = plt.subplots(figsize=(6, 2.5))
    im = ax.imshow(attention_2d, cmap='jet', aspect='equal', interpolation='nearest')

    # ✅ 添加氨基酸字母
    for i, aa in enumerate(sequence):
        ax.text(i, -0.8, aa, ha='center', va='center', fontsize=10, family='monospace')

    # ✅ 设置刻度（注意共 23 个残基）
    tick_pos = np.arange(0, len(sequence), 5)
    tick_labels = np.arange(121, 144, 5)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)

    ax.set_yticks([])
    ax.set_xlabel('Residue Index')
    ax.set_title(title, pad=10)

    # 添加 colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad=0.02)
    cbar.set_label('Normalized Attention')

    plt.tight_layout()

    # 如需保存请取消注释
    plt.savefig(save_path, format='eps', dpi=300)

    plt.show()



model.load_state_dict(torch.load('trained_model.pt'))
for _p in ['re']:
    y_hat,att2,att4,attn_merged = test(model, data_loaders[_p], device, SHOW_PROCESS_BAR, _p)
    sequence_str = "RTGYDNREIVMKYIHYKLSQRGYEWDAGSEVVHLTLRQAGDDFSRRYRRDFAEMSSQLHLTPFTARGRFATVVEELFRDGVNWGRIVAFFEFGGVMCVESVNREMSPLVDNIALWMTEYLNRHLHTWIQDNGGWDAFVELYGP"
    # att2_crop = att2[0, :143, :92]
    avg_attention = attn_merged[0, :143]
    # avg_attention = att2_crop.mean(dim=1)
    # avg_attention = avg_attention[:143]
    min_val = avg_attention.min()
    max_val = avg_attention.max()
    if max_val > min_val:
        avg_attention_norm = (avg_attention - min_val) / (max_val - min_val)
    else:
        avg_attention_norm = np.zeros_like(avg_attention)

    avg_attention_norm = avg_attention_norm.cpu().numpy()

    plot_1d_attention_heatmap(avg_attention_norm, sequence_str, save_path='120-143.eps')

    # plot_attention_paper_style(sequence_str, avg_attention_norm, title='Head2 Residue-wise Average Attention')


print('training finished')

end = datetime.now()
print('end at:', end)
print('time used:', str(end - start))
