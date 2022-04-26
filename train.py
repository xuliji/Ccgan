from comet_ml import Experiment
from tqdm import tqdm
from modules import Generator, Discriminator
from dataset import MyDataset
import torch
import numpy as np
from torch.autograd import Variable

experiment = Experiment(
    api_key="93lqYGCEkdfNVApUyjBxYCE2e",
    project_name="ccgan",
    workspace="xuliji",
)


# Loss function
adversarial_loss = torch.nn.MSELoss()


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

cuda = True if torch.cuda.is_available() else False

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()


# load data
dataset = MyDataset(shapes_dir='./picked_uiuc', target_dir='./result/07', names_txt='./picked_uiuc_list.txt', alpha=1.0)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True)

lr = 0.0002
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


n_epochs = 20000
step = 0
for epoch in range(n_epochs):
    for i, (labels, shapes) in enumerate(dataloader):
        batch_size = shapes.shape[0]
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        # real labels
        real_imgs = Variable(shapes.type(FloatTensor))
        labels = Variable(labels.type(FloatTensor))


        # Train Generator
        optimizer_G.zero_grad()
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, 100))))
        gen_labels = Variable(FloatTensor(labels))
        gen_shapes = generator(z, gen_labels)
        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_shapes, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        validity_fake = discriminator(gen_shapes.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        step += 1
        experiment.log_metric("d_loss", d_loss.item(), step=step)
        experiment.log_metric("g_loss", g_loss.item(), step=step)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

torch.save(generator.state_dict(), './generator.pth')
experiment.end()

