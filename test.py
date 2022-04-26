from modules import Generator
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable


cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

model = Generator()
model.load_state_dict(torch.load('./generator.pth'))
model.to(torch.device('cuda'))
model.eval()
target = np.load('./result/07/a18.npy', allow_pickle=True)[42, :].reshape(1, 5)

target = torch.from_numpy(target).float().to(torch.device('cuda'))

z = Variable(FloatTensor(np.random.normal(0, 1, (1, 100))))

shape = model(z, target)
shape = shape.cpu().detach().numpy().reshape(201, 2)
plt.figure(figsize=(10, 10))
plt.xlim(0, 1)
plt.ylim(-0.5, 0.5)
plt.scatter(x=shape[:, 0], y=shape[:, 1], s=20, c='r')
plt.savefig('./a18.png')
plt.show()





