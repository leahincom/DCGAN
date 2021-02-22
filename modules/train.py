from model import *
from dataset import *

import torch
import torch

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from statistics import mean


class Train:
    def __init__(self, args):
        self.mode = args.mode
        self.train_continue = args.train_continue

        self.scope = args.scope
        self.dir_checkpoint = args.dir_checkpoint
        self.dir_log = args.dir_log

        self.dir_data = args.dir_data
        self.dir_result = args.dir_result

        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size

        self.lr_G = args.lr_G
        self.lr_D = args.lr_D

        self.optim = args.optim
        self.beta1 = args.beta1

        self.ny_in = args.ny_in
        self.nx_in = args.nx_in
        self.nch_in = args.nch_in

        self.ny_load = args.ny_load
        self.nx_load = args.nx_load
        self.nch_load = args.nch_load

        self.ny_out = args.ny_out
        self.nx_out = args.nx_out
        self.nch_out = args.nch_out

        self.nch_ker = args.nch_ker

        self.data_type = args.data_type
        self.norm = args.norm

        self.gpu_ids = args.gpu_ids

        self.num_freq_disp = args.num_freq_disp
        self.num_freq_save = args.num_freq_save

        self.name_data = args.name_data

        if self.gpu_ids and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % self.gpu_ids[0])
            torch.cuda.set_device(self.gpu_ids[0])
        else:
            self.device = torch.device('cpu')

    def save(self, dir_chck, netG, netD, optimG, optimD, epoch):
        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        torch.save({'netG': netG.state_dict(), 'netD': netD.state_dict(),
                    'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
                   '%s/model_epoch%04d.pth' % (dir_chck, epoch))

    def load(self, dir_chck, netG, netD=[], optimG=[], optimD=[], epoch=[], mode='train'):
        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

        print('Loaded %dth network' % epoch)

        if mode == 'train':
            netG.load_state_dict(dict_net['netG'])
            netD.load_state_dict(dict_net['netD'])
            optimG.load_state_dict(dict_net['optimG'])
            optimD.load_state_dict(dict_net['optimD'])

            return netG, netD, optimG, optimD, epoch

        # test the efficiency of generator
        elif mode == 'test':
            netG.load_state_dict(dict_net['netG'])

            return netG, epoch

    def preprocess(self, data):
        rescale = Rescale((self.ny_load, self.nx_load))
        randomcrop = Randomcrop((self.ny_out, self.nx_out))
        normalize = Normalize()
        randomflip = RandomFlip()
        totensor = ToTensor()

        return totensor(normalize(rescale(data)))

    def deprocess(self, data):
        tonumpy = ToNumpy()
        denomalize = Denomalize()

        return denomalize(tonumpy(data))

    def train(self):
        mode = self.mode

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr_G = self.lr_G
        lr_D = self.lr_D

        wgt_gan = self.wgt_gan
        wgt_disc = self.wgt_disc

        batch_size = self.batch_size
        device = self.device

        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        norm = self.norm
        name_data = self.name_data

        num_freq_disp = self.num_freq_disp
        num_freq_save = self.num_freq_save

        ny_in = self.ny_in
        nx_in = self.nx_in

        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_data_train = os.path.join(self.dir_data, name_data)
        dir_log = os.path.join(self.dir_log, self.scope, name_data)

        transform_train = transforms.Compose(
            [Normalize(), Rescale((self.ny_load, self.nx_load)), ToTensor()])
        transform_inv = transforms.Compose([ToNumpy(), Denomalize()])

        dataset.train = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

        num_train = len(dataset_train)
        num_batch_train = int((num_train / batch_size) +
                              ((num_train % batch_size) != 0))

        netG = Generator(nch_in, nch_out, nch_ker, norm)
        netD = Discriminator(nch_out, nch_ker, norm)

        init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(netD, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        fn_GAN = nn.BCEWithLogitsLoss().to(device)

        paramsG = netG.parameters()
        paramsD = netD.parameters()

        optimG = torch.optim.Adam(paramsG, lr=lr_G, betas=(self.bet1, 0.999))
        optimD = torch.optim.Adam(paramsD, lr=lr_D, betas=(self.bet1, 0.999))

        st_epoch = 0

        if train_continue == 'on':
            netG, netD, optimG, optimD, st_epoch = self.load(
                dir_chck, netG, netD, optimG, optimD, mode=mode)

            writer_train = SummaryWriter(log_dir=dir_log)

            for epoch in range(st_epoch+1, num_epoch+1):
                netG.train()
                netD.train()

                loss_G_train = []
                loss_D_real_train = []
                loss_D_fake_train = []

                for i, data in enumerate(loader_train, 1):
                    def should(freq):
                        return freq > 0 and (i % freq == 0 or i == num_batch_train)

                    label = data.to(device)
                    input = torch.randn(label.size(
                        0), nch_in, ny_in, nx_in).to(device)

                    output = netG(input)

                    set_requires_grad(netD, True)
                    optimD.zero_grad

                    pred_real = netD(label)
                    pred_fake = netD(output.detach())

                    loss_D_real = fn_GAN(pred_real, torch.ones_like(pred_real))
                    loss_D_fake = fn_GAN(
                        pred_fake, torch.zeros_like(pred_fake))
                    loss_D = 0.5 * (loss_D_real + loss_D_fake)

                    loss_D.backward()
                    optimD.step()

                    set_requires_grad(netD, False)
                    optimG.zero_grad()

                    pred_fake = netD(output)

                    loss_G = fn_GAN(pred_fake, torch.ones_like(pred_fake))

                    loss_G.backward()
                    optimG.step()

                    loss_G_train += [loss_G.item()]
                    loss_D_real_train += [loss_D_real.item()]
                    loss_D_fake_train += [loss_D_fake.item()]

                    print('TRAIN: EPOCH %d: BATCH %04d/%04d: '
                          'GEN GAN: %.4f DISC FAKE: %.4f DISC REAL: %.4f' %
                          (epoch, i, num_batch_train,
                           mean(loss_G_train), mean(loss_D_fake_train), mean(loss_D_real_train)))

                    if should(num_freq_disp):
                        output = transform_inv(output)
                        label = transform_inv(label)

                        writer_train.add_images(
                            'output', output, num_batch_train*(epoch-1)+i, dataformats='NHWC')
                        writer_train.add_images(
                            'label', label, num_batch_train*(epoch-1)+i, dataformats='NHWC')

                writer_train.add_scalar(
                    'loss_G', mean(loss_G_train), epoch)
                writer_train.add_scalar(
                    'loss_D_fake', mean(loss_D_fake_train), epoch)
                writer_train.add_scalar(
                    'loss_D_real', mean(loss_D_real_train), epoch)

                if (epoch % num_freq_save) == 0:
                    self.save(dir_chck, netG, netD, optimG, optimD, epoch)

        writer_train.close()

    def test(self):
        mode = self.mode

        batch_size = self.batch_size
        device = self.device
        gpu_ids = self.gpu_ids

        ny_in = self.ny_in
        nx_in = self.nx_in

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        norm = self.norm

        name_data = self.name_data

        dir_chck = os.path.join(
            self.dir_checkpoint, self.scope, name_data)

        dir_result = os.path.join(
            self.dir_result, self.scope, name_data)
        dir_result_save = os.path.join(dir_result, 'images')
        if not os.path.exists(dir_result_save):
            os.makedirs(dir_result_save)

        transform_inv = transforms.Compose([ToNumpy(), Denomalize()])

        netG = DCGAN(nch_in, nch_out, nch_ker, norm)
        init_net(netG, init_type='normal',
                 init_gain=0.02, gpu_ids=gpu_ids)

        st_epoch = 0

        netG, st_epoch = self.load(dir_chck, netG, mode=mode)

        with torch.no_grad():
            netG.eval()

            input = torch.randn(batch_size, nch_in, ny_in, nx_in).to(device)
            output = netG(input)
            output = transform_inv(output)

            for j in range(output.shape[0]):
                name = j
                fileset = {'name': name,
                           'output': "%04d-output.png" % name}

                if nch_out == 3:
                    plt.imsave(os.path.join(dir_result_save,
                                            fileset['output']), output[j, :, :, :].squeeze())
                elif nch_out == 1:
                    plt.imsave(os.path.join(
                        dir_result_save, fileset['output']), output[j, :, :, :].squeeze(), cmap=cm.gray)

                append_index(dir_result, fileset)

    def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]

        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def append_index(dir_result, fileset, step=False):
        index_path = os.path.join(dir_result, "index.html")
        if os.path.exists(index_path):
            index = open(index_path, "a")
        else:
            index = open(index_path, "w")
            index.write("<html><body><table><tr>")
            if step:
                index.write("<th>step</th>")
            for key, value in fileset.items():
                index.write("<th>%s</th>" % key)
            index.write('</tr>')

        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        del fileset['name']

        for key, value in fileset.items():
            index.write("<td><img src='images/%s'></td>" % value)

        index.write("</tr>")
        return index_path
