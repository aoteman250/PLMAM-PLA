import sys
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dataset import *
from model import MultiViewNet
import os
import metrics
import torch
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print(sys.argv)

SHOW_PROCESS_BAR = True
data_path = '../data/'
device = torch.device("cuda")

batch_size = 6

model = MultiViewNet()

model = model.to(device)

data_loaders = {phase_name:
                    DataLoader(MyDataset(data_path, phase_name,
                                         max_smi_len=120, max_seq_len=1000),
                               batch_size=batch_size,
                               shuffle=False,
                               )

                for phase_name in ['training', 'validation', 'test']}


loss_function = nn.MSELoss(reduction='sum')  #
start = datetime.now()
print('start at ', start)

def test(model: nn.Module, test_loader, loss_function, device, show, _p):
    path = '../script/result-A/'
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for idx, (*x, y) in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):
            smile = x[0].to(device)
            sequence = x[1].to(device)
            smi = x[2].to(device)
            seq = x[3].to(device)
            y = y.to(device)
            y_hat = model(smile,sequence ,smi,seq )
            test_loss += loss_function(y_hat.view(-1), y.view(-1)).item()
            outputs.append(y_hat.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)

    np.savetxt(path + _p  + '-targets.csv', targets)
    np.savetxt(path + _p  + '-outputs.csv', outputs)

    test_loss /= len(test_loader.dataset)

    evaluation = {
        'loss': test_loss,
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),
        'pearson': metrics.get_pearson(targets, outputs),
    }

    return evaluation

model.load_state_dict(torch.load('best_model.pt'))
for _p in ['test']:
    performance = test(model, data_loaders[_p], loss_function, device, SHOW_PROCESS_BAR, _p)
    print(performance)

    print(f'{_p}:')
    for k, v in performance.items():
        print(f'{k}: {v}')
    print()

print('training finished')

end = datetime.now()
print('end at:', end)
print('time used:', str(end - start))
