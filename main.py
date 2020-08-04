from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import argparse
import os
import numpy as np
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import classification_report
import copy
import random as ran
import torch.nn.functional
from utils import weights_init, compute_acc, fitness_score, show_result
from network import _netG_28, _netD_28

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='mnist')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--imageChannel', type=int, default=1, help='the channel of the input image to network')
parser.add_argument('--nz', type=int, default=102, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--T', type=float, default=1000, help='initial temperature. default=1000')
parser.add_argument('--alpha', type=float, default=0.99, help='annealing coefficient. default=0.99')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes for AG-GAN')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')

opt = parser.parse_args()
file_name = open('result.txt', 'a')
print(opt)
print(opt, file=file_name)

# specify the gpu id if using only 1 gpu
if opt.ngpu == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed, file=file_name)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# datase t
# the training set has been processed by imbalanced sampling
image = np.loadtxt('mnist_image.txt')
label = np.loadtxt('mnist_label.txt')
image = image.reshape((-1, 1, 28, 28))
# the test set is balanced dataset
test_image = np.loadtxt('test_image.txt')
test_label = np.loadtxt('test_label.txt')
test_image = test_image.reshape((-1, 1, 28, 28))


image = torch.from_numpy(image)
label = torch.from_numpy(label)
test_image = torch.from_numpy(test_image)
test_label = torch.from_numpy(test_label)
dataset = torch.utils.data.TensorDataset(image, label)
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

# some hyper parameters
ngpu = int(opt.ngpu)
nz = int(opt.nz)
num_classes = int(opt.num_classes)

# Define the generator and initialize the weights
if opt.dataset == 'mnist':
    netG = _netG_28(ngpu, nz)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))

# Define the discriminator and initialize the weights
if opt.dataset == 'mnist':
    netD = _netD_28(ngpu, num_classes)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))

# loss functions
ls_loss = nn.MSELoss()
van_loss = nn.BCELoss()
loss_type = ['trickLogD', 'minimax', 'ls']
dis_criterion = nn.BCELoss()
aux_criterion = nn.NLLLoss()

# tensor placeholders
input = torch.FloatTensor(opt.batchSize, opt.imageChannel, opt.imageSize, opt.imageSize)
test_input = torch.FloatTensor(len(test_label), opt.imageChannel, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
eval_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
disr_label = torch.FloatTensor(opt.batchSize)
disf_label = torch.FloatTensor(opt.batchSize)
tes_label = torch.FloatTensor(len(test_label))
aux_label = torch.LongTensor(opt.batchSize)
real_label = 1
fake_label = 0

# if using cuda
if opt.cuda:
    netD.cuda()
    netG.cuda()
    ls_loss.cuda()
    van_loss.cuda()
    aux_criterion.cuda()
    input, disr_label, disf_label, aux_label = input.cuda(), disr_label.cuda(), disf_label.cuda(), aux_label.cuda()
    test_input, test_image, tes_label = test_input.cuda(), test_image.cuda(), tes_label.cuda()
    noise, eval_noise = noise.cuda(), eval_noise.cuda()

# define variables
input = Variable(input)
test_input = Variable(test_input)
noise = Variable(noise)
eval_noise = Variable(eval_noise)
disr_label = Variable(disr_label)
disf_label = Variable(disf_label)
tes_label = Variable(tes_label)
aux_label = Variable(aux_label)

# noise for evaluation
eval_noise_ = np.random.normal(0, 1, (opt.batchSize, nz))
eval_label = np.random.randint(0, num_classes, opt.batchSize)
eval_onehot = np.zeros((opt.batchSize, num_classes))
eval_onehot[np.arange(opt.batchSize), eval_label] = 1
eval_noise_[np.arange(opt.batchSize), :num_classes] = eval_onehot[np.arange(opt.batchSize)]
eval_noise_ = (torch.from_numpy(eval_noise_))
eval_noise.data.copy_(eval_noise_.view(opt.batchSize, nz, 1, 1))

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

with torch.no_grad():
    test_input.resize_(test_image.size()).copy_(test_image)
    tes_label.resize_(len(test_label)).copy_(test_label)
accuracy_max = 0
for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ###########################
        # (1) Update D network:
        ###########################
        for _ in range(1):
            # train with real
            netD.zero_grad()
            real_cpu, label = data
            batch_size = real_cpu.size(0)
            if opt.cuda:
                real_cpu = real_cpu.cuda()
            with torch.no_grad():
                input.resize_(real_cpu.size()).copy_(real_cpu)
                disr_label.resize_(batch_size).fill_(real_label)
                aux_label.resize_(batch_size).copy_(label)
            dis_output, aux_output = netD(input)
            dis_errD_real = van_loss(dis_output, disr_label)
            aux_errD_real = aux_criterion(aux_output, aux_label)
            errD_real = dis_errD_real + aux_errD_real
            errD_real.backward()
            D_x = dis_output.data.mean()
            # compute the current classification accuracy
            accuracy = compute_acc(aux_output, aux_label)

            # train with fake
            with torch.no_grad():
                noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
            label = np.random.randint(0, num_classes, batch_size)
            noise_ = np.random.normal(0, 1, (batch_size, nz))
            class_onehot = np.zeros((batch_size, num_classes))
            class_onehot[np.arange(batch_size), label] = 1
            noise_[np.arange(batch_size), :num_classes] = class_onehot[np.arange(batch_size)]
            noise_ = (torch.from_numpy(noise_))
            noise.data.copy_(noise_.view(batch_size, nz, 1, 1))
            aux_label.resize_(batch_size).copy_(torch.from_numpy(label))

            fake = netG(noise)
            dis_output, aux_output = netD(fake.detach())
            disf_label.resize_(batch_size).fill_(fake_label)
            dis_errD_fake = van_loss(dis_output, disf_label)
            aux_errD_fake = aux_criterion(aux_output, aux_label)
            errD_fake = dis_errD_fake + aux_errD_fake
            errD_fake.backward()
            D_G_z1 = dis_output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()



        ###########################
        # (2) Update G network:
        ###########################
        for _ in range(1):
            # Sample noise as generator input
            # variation-evluation-selection
            # netD.requires_grad=False

            G_list = copy.deepcopy(netG.state_dict())
            optG_list = copy.deepcopy(optimizerG.state_dict())

            with torch.no_grad():
                fake = netG(noise)
            dis_output_r, aux_output_r = netD(input)
            dis_output_f, aux_output_f = netD(fake.detach())
            # Adversarial loss
            errD_real = van_loss(dis_output_r, disr_label) + aux_criterion(aux_output_r, aux_label)
            errD_fake = van_loss(dis_output_f, disf_label) + aux_criterion(aux_output_f, aux_label)
            errD = errD_real + errD_fake
            F, Fq, Fd = fitness_score(netD, errD, errD_fake)

            for j in range(3):
                # Variation
                # netG.load_state_dict(G_list)
                # optimizerG.load_state_dict(optG_list)
                netG.zero_grad()
                fake = netG(noise)
                dis_output, aux_output = netD(fake)
                if j == 0:
                    dis_errG = van_loss(dis_output, disr_label)
                elif j == 1:
                    dis_errG = -van_loss(dis_output, disf_label)
                elif j == 2:
                    dis_errG = ls_loss(dis_output, disr_label)
                aux_errG = aux_criterion(aux_output, aux_label)
                errGc = dis_errG + aux_errG
                errGc.backward()
                D_G_z2 = dis_output.data.mean()
                optimizerG.step()

                fake = netG(noise)
                dis_output_r, aux_output_r = netD(input)
                dis_output_f, aux_output_f = netD(fake.detach())
                # Adversarial loss
                errD_real = van_loss(dis_output_r, disr_label) + aux_criterion(aux_output_r, aux_label)
                errD_fake = van_loss(dis_output_f, disf_label) + aux_criterion(aux_output_f, aux_label)
                errD = errD_real + errD_fake
                F_c, Fq_c, Fd_c = fitness_score(netD, errD, errD_fake)

                # Selection
                if j < 1:
                    F_flag = F_c
                    G_c_list = copy.deepcopy(netG.state_dict())
                    optG_c_list = copy.deepcopy(optimizerG.state_dict())
                    errGc_ = errGc
                    type = loss_type[j]
                else:
                    fit_com = F_c - F_flag
                    if fit_com > 0:
                        F_flag = F_c
                        G_c_list = copy.deepcopy(netG.state_dict())
                        optG_c_list = copy.deepcopy(optimizerG.state_dict())
                        errGc_ = errGc
                        type = loss_type[j]
            fit_com_G = F_flag - F
            if fit_com_G > 0:
                G_list = copy.deepcopy(G_c_list)
                optG_list = copy.deepcopy(optG_c_list)
                errG = errGc_
            else:
                p = np.exp(fit_com_G / opt.T)
                u = ran.uniform(0, 1)
                if u < p:
                    G_list = copy.deepcopy(G_c_list)
                    optG_list = copy.deepcopy(optG_c_list)
                    errG = errGc_

            netG.load_state_dict(G_list)
            optimizerG.load_state_dict(optG_list)

        opt.T = opt.alpha * opt.T

        ###########################
        # test
        ###########################
        if i % 100 == 0:
            _, test_output = netD(test_input)
            accuracy_t = compute_acc(test_output, tes_label)
            if accuracy_t > accuracy_max:
                accuracy_max = accuracy_t
                preds = test_output.data.max(1)[1]
                print(classification_report(tes_label.cpu().detach().numpy(), preds.cpu().detach().numpy(), digits=6),
                      file=file_name)
                torch.save(netG.state_dict(), './netG_max.pth')
                torch.save(netD.state_dict(), './netD_max.pth')
            print(
                '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Acc: %.4f AccT: %.4f MAX: %.4f'
                % (epoch + 1, opt.niter, i + 1, len(dataloader),
                   errD, errG, D_x, D_G_z1, D_G_z2, accuracy, accuracy_t, accuracy_max), file=file_name)

    ###########################
    # results visualization & model preservation
    ###########################
    if (epoch + 1) % 50 == 0:
        show_result(real_cpu.cpu().detach().numpy(), (epoch + 1), path='%s/real_samples.png' % opt.outf)
        fake = netG(eval_noise)
        show_result(fake.cpu().detach().numpy(), (epoch + 1),
                    path='%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch + 1))
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch + 1))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch + 1))

