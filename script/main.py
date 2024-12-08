from datetime import datetime
from torch import _pin_memory, nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dataset import *
from model import MultiViewNet, test
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
SHOW_PROCESS_BAR = True
data_path = '../data/'
seed = np.random.randint(2023,2024)
device = torch.device("cuda")
batch_size = 6
n_epoch = 20
save_best_epoch = 5


torch.backends.cudnn.deterministic = False 
torch.backends.cudnn.benchmark =  True
torch.backends.cudnn.enable = True
torch.manual_seed(seed)
np.random.seed(seed)
model = MultiViewNet()
model = model.to(device)

data_loaders = {phase_name:
                    DataLoader(MyDataset(data_path, phase_name,
                                         max_smi_len=120,max_seq_len=1000),
                               batch_size=batch_size,
                               shuffle= True,
                               )

                for phase_name in ['training', 'validation', 'test']}

optimizer = optim.AdamW(model.parameters() ,lr=0.00005)
loss_function = nn.MSELoss(reduction='sum')
start = datetime.now()
print('start at ', start)
best_epoch = -1
best_val_loss = 100000000

for epoch in range(1, n_epoch + 1):
    total_loss = 0.0
    tbar = tqdm(enumerate(data_loaders['training']), disable=not SHOW_PROCESS_BAR, total=len(data_loaders['training']))
    for idx, (*x, y) in tbar:
        model.train()
        smile = x[0].to(device)
        sequence = x[1].to(device)
        smi = x[2].to(device)
        seq = x[3].to(device)

        y = y.to(device)
        optimizer.zero_grad()

        output = model(smile,sequence,smi,seq )
        loss = loss_function(output.view(-1), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()  # Accumulate the loss value for the current batch
        tbar.set_description(f' * Train Epoch {epoch} Loss={loss.item() / len(y):.3f}')

    # Prints the total loss value for the current epoch
    avg_loss = total_loss / len(data_loaders['training'])
    print(f'Epoch {epoch}, Total Loss: {avg_loss:.3f}')

    for _p in ['validation','test']:
        performance = test(model, data_loaders[_p], loss_function, device, False, _p)
        print('epoch:',epoch)
        print(f'{_p}:')
        for k, v in performance.items():
            print(f'{k}: {v}')

        if _p=='validation' and epoch>=save_best_epoch and performance['loss']<best_val_loss:
            best_val_loss = performance['loss']
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pt')
    
model.load_state_dict(torch.load('best_model.pt'))
for _p in ['test']:
    performance = test(model, data_loaders[_p], loss_function, device, SHOW_PROCESS_BAR, _p)

    print(f'{_p}:')
    for k, v in performance.items():
        print(f'{k}: {v}')
    print()


print('training finished')
end = datetime.now()
print('end at:', end)
print('time used:', str(end - start))
