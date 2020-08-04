import matplotlib
matplotlib.use('Agg')
import torch.nn.functional
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# compute the current classification accuracy
def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data.long()).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc


def fitness_score(netD,errD,errD_fake):
    lamda = 0.2
    netD.zero_grad()
    netD.requires_grad = True
    # Quality fitness score
    Fq = errD_fake.data.mean().cpu().detach().numpy()

    # Diversity fitness score
    gradients = torch.autograd.grad(outputs=errD, inputs=netD.parameters(),
                                    grad_outputs=torch.ones(errD.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)
    with torch.no_grad():
        for i, grad in enumerate(gradients):
            grad = grad.view(-1)
            allgrad = grad if i == 0 else torch.cat([allgrad,grad])
    Fd = torch.log(torch.norm(allgrad)).data.cpu().detach().numpy()
    F = Fq + lamda * Fd
    return F, Fq, Fd


def show_result(test_images,num_epoch, show = False, path = 'result.png'):

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5)) #nrows,ncols
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)): #创建笛卡尔积
        ax[i, j].get_xaxis().set_visible(False) #不显示坐标轴
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow(np.clip(np.reshape(test_images[k], (128, 128,3)),0,1))

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()