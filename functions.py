# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from imageio import imsave
from tqdm import tqdm
from copy import deepcopy
import logging
import ot
from utils.inception_score import get_inception_score
from utils.fid_score import calculate_fid_given_paths

logger = logging.getLogger(__name__)


def train(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch,
          writer_dict, schedulers=None):
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()

    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        true_bs = imgs.shape[0]
        if (true_bs < args.gen_batch_size):
            break
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size()

        fake_validity = dis_net(fake_imgs)

        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        d_loss.backward()
        dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), global_steps)

        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            gen_imgs = gen_net(gen_z)
            fake_validity = dis_net(gen_imgs)

            # cal loss
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1
        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1

def train_sw(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch,
          writer_dict, schedulers=None):
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()

    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        true_bs = imgs.shape[0]
        if (true_bs < args.gen_batch_size):
            break
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)
        true_bs = imgs.shape[0]
        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (true_bs, args.latent_dim)))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size()

        fake_validity = dis_net(fake_imgs)
        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        d_loss.backward()
        dis_optimizer.step()
        writer.add_scalar('d_loss', d_loss.item(), global_steps)
        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            real_validity, real_features = dis_net(real_imgs, return_feature=True)
            gen_optimizer.zero_grad()
            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (true_bs, args.latent_dim)))
            gen_imgs = gen_net(gen_z)
            fake_validity, fake_features = dis_net(gen_imgs, return_feature=True)
            # cal loss
            X = torch.cat([real_features[-1].view(true_bs, -1),real_validity.view(true_bs,-1)],dim=1)
            Y = torch.cat([fake_features[-1].view(true_bs, -1),fake_validity.view(true_bs,-1)],dim=1)
            # cal loss
            g_loss =  SW(X,Y,L=args.L)

            g_loss.backward()
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1

def train_maxsw(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch,
          writer_dict, schedulers=None):
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()

    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        true_bs = imgs.shape[0]
        if (true_bs < args.gen_batch_size):
            break
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)
        true_bs = imgs.shape[0]
        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (true_bs, args.latent_dim)))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size()

        fake_validity = dis_net(fake_imgs)
        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        d_loss.backward()
        dis_optimizer.step()
        writer.add_scalar('d_loss', d_loss.item(), global_steps)
        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            real_validity, real_features = dis_net(real_imgs, return_feature=True)
            gen_optimizer.zero_grad()
            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (true_bs, args.latent_dim)))
            gen_imgs = gen_net(gen_z)
            fake_validity,fake_features = dis_net(gen_imgs,return_feature=True)
            # cal loss
            X = torch.cat([real_features[-1].view(true_bs, -1), real_validity.view(true_bs, -1)], dim=1)
            Y = torch.cat([fake_features[-1].view(true_bs, -1), fake_validity.view(true_bs, -1)], dim=1)
            g_loss,theta = maxSW(X,Y,max_iter=args.s_max_iter,lr=args.s_lr)
            g_loss = g_loss
            g_loss.backward()
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1
def train_prw(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch,
          writer_dict, schedulers=None):
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()

    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        true_bs = imgs.shape[0]
        if (true_bs < args.gen_batch_size):
            break
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)
        true_bs = imgs.shape[0]
        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (true_bs, args.latent_dim)))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size()

        fake_validity = dis_net(fake_imgs)
        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        d_loss.backward()
        dis_optimizer.step()
        writer.add_scalar('d_loss', d_loss.item(), global_steps)
        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            real_validity, real_features = dis_net(real_imgs, return_feature=True)
            gen_optimizer.zero_grad()
            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (true_bs, args.latent_dim)))
            gen_imgs = gen_net(gen_z)
            fake_validity,fake_features = dis_net(gen_imgs,return_feature=True)
            # cal loss
            X = torch.cat([real_features[-1].view(true_bs, -1), real_validity.view(true_bs, -1)], dim=1)
            Y = torch.cat([fake_features[-1].view(true_bs, -1), fake_validity.view(true_bs, -1)], dim=1)
            g_loss,theta = PRW(X,Y,args.k,max_iter=args.s_max_iter,lr=args.s_lr)
            g_loss = g_loss
            g_loss.backward()
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1
def train_amortizedsw(args, gen_net: nn.Module, dis_net: nn.Module,f_slice:nn.Module, gen_optimizer, dis_optimizer,f_optimizer, gen_avg_param, train_loader, epoch,
          writer_dict, schedulers=None):
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()
    f_slice = f_slice.train()

    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        true_bs = imgs.shape[0]
        if(true_bs <args.gen_batch_size):
            break
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (true_bs, args.latent_dim)))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size()

        fake_validity = dis_net(fake_imgs)
        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        d_loss.backward()
        dis_optimizer.step()
        writer.add_scalar('d_loss', d_loss.item(), global_steps)
        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            real_validity, real_features = dis_net(real_imgs, return_feature=True)

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (true_bs, args.latent_dim)))
            gen_imgs = gen_net(gen_z)
            fake_validity, fake_features = dis_net(gen_imgs, return_feature=True)
            # cal loss
            X = torch.cat([real_features[-1].view(true_bs, -1), real_validity.view(true_bs, -1)], dim=1)
            Y = torch.cat([fake_features[-1].view(true_bs, -1), fake_validity.view(true_bs, -1)], dim=1)
            # cal loss
            g_loss,theta = amortized_maxSW(X.detach(),Y.detach(),f_slice,detach=args.detach)
            f_optimizer.zero_grad()
            (-g_loss).backward()
            f_optimizer.step()
            gen_optimizer.zero_grad()
            g_loss,theta = amortized_maxSW(X,Y, f_slice,detach=args.detach)
            g_loss = g_loss
            g_loss.backward()
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler, f_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                s_lr = f_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)
                writer.add_scalar('LR/s_lr', s_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1
def train_amortizedprw(args, gen_net: nn.Module, dis_net: nn.Module,f_slice:nn.Module, gen_optimizer, dis_optimizer,f_optimizer, gen_avg_param, train_loader, epoch,
          writer_dict, schedulers=None):
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()
    f_slice = f_slice.train()

    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        true_bs = imgs.shape[0]
        if(true_bs <args.gen_batch_size):
            break
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (true_bs, args.latent_dim)))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size()

        fake_validity = dis_net(fake_imgs)
        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        d_loss.backward()
        dis_optimizer.step()
        writer.add_scalar('d_loss', d_loss.item(), global_steps)
        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            real_validity, real_features = dis_net(real_imgs, return_feature=True)

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (true_bs, args.latent_dim)))
            gen_imgs = gen_net(gen_z)
            fake_validity, fake_features = dis_net(gen_imgs, return_feature=True)
            # cal loss
            X = torch.cat([real_features[-1].view(true_bs, -1), real_validity.view(true_bs, -1)], dim=1)
            Y = torch.cat([fake_features[-1].view(true_bs, -1), fake_validity.view(true_bs, -1)], dim=1)
            # cal loss
            g_loss,theta = amortized_PRW(X.detach(),Y.detach(),f_slice,detach=args.detach)
            f_optimizer.zero_grad()
            (-g_loss).backward()
            f_optimizer.step()
            gen_optimizer.zero_grad()
            g_loss,theta = amortized_PRW(X,Y, f_slice,detach=args.detach)
            g_loss = g_loss
            g_loss.backward()
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler, f_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                s_lr = f_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)
                writer.add_scalar('LR/s_lr', s_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1
def validate(args, fixed_z, fid_stat, gen_net: nn.Module, writer_dict):
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']

    # eval mode
    gen_net = gen_net.eval()

    # generate images
    sample_imgs = gen_net(fixed_z)
    img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)

    # get fid and inception score
    fid_buffer_dir = os.path.join(args.path_helper['sample_path'], 'fid_buffer')
    os.makedirs(fid_buffer_dir)

    eval_iter = args.num_eval_imgs // args.eval_batch_size
    img_list = list()
    for iter_idx in tqdm(range(eval_iter), desc='sample images'):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

        # Generate a batch of images
        gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        for img_idx, img in enumerate(gen_imgs):
            file_name = os.path.join(fid_buffer_dir, f'iter{iter_idx}_b{img_idx}.png')
            imsave(file_name, img)
        img_list.extend(list(gen_imgs))

    # get inception score
    logger.info('=> calculate inception score')
    if args.dataset.lower() == 'lsun_church' or args.dataset.lower() == 'celebahq' :
        mean, std = get_inception_score(img_list,bs=16)
    elif args.dataset.lower() == 'stl10':
        mean, std = get_inception_score(img_list,bs=32)
    else:
        mean, std = get_inception_score(img_list,bs=100)

    # get fid score
    logger.info('=> calculate fid score')
    if args.dataset.lower() == 'lsun_church' or args.dataset.lower() == 'celebahq' :
        fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None,batch_size=args.eval_batch_size)
    elif args.dataset.lower() == 'stl10':
        fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None,batch_size=args.eval_batch_size)
    else:
        fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None,batch_size=args.eval_batch_size)
    os.system('rm -r {}'.format(fid_buffer_dir))

    writer.add_image('sampled_images', img_grid, global_steps)
    writer.add_scalar('Inception_score/mean', mean, global_steps)
    writer.add_scalar('Inception_score/std', std, global_steps)
    writer.add_scalar('FID_score', fid_score, global_steps)

    writer_dict['valid_global_steps'] = global_steps + 1

    return mean, fid_score


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections

def one_dimensional_Wasserstein(X,Y,theta,p):
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    wasserstein_distance = torch.abs(
        (
                torch.sort(X_prod.transpose(0, 1), dim=1)[0]
                - torch.sort(Y_prod.transpose(0, 1), dim=1)[0]
        )
    )
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1.0 / p)
    wasserstein_distance=torch.pow(torch.pow(wasserstein_distance, p).mean(), 1.0 / p)
    return wasserstein_distance
def k_dimensional_Wasserstein(X,Y,U,p,device="cuda"):
    X_prod = torch.matmul(X, U)
    Y_prod = torch.matmul(Y, U)
    M  = torch.cdist(X_prod, Y_prod,p=p)**p
    a,b = ot.unif(X_prod.shape[0]), ot.unif(Y_prod.shape[0])
    plan = torch.from_numpy(ot.emd(a,b,M.detach().cpu().numpy())).to(device)
    wasserstein_distance = torch.pow(torch.sum(plan*M), 1.0 / p)
    return wasserstein_distance
def maxSW(X, Y, p=2, max_iter=100, lr=1e-2,device="cuda"):
    theta = torch.randn((1, X.shape[1]), device=device, requires_grad=True)
    theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1))
    opt = torch.optim.Adam([theta], lr=lr)
    Xdetach = X.detach()
    Ydetach = Y.detach()
    for _ in range(max_iter):
        l = -one_dimensional_Wasserstein(Xdetach,Ydetach,theta,p)
        opt.zero_grad()
        l.backward()
        opt.step()
        theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1))
    wasserstein_distance=one_dimensional_Wasserstein(X,Y,theta,p)
    return wasserstein_distance, theta

def SW(X, Y, L=1000, p=2, device="cuda"):
    dim = X.size(1)
    theta = rand_projections(dim, L).to(device)
    return  one_dimensional_Wasserstein(X,Y,theta,p)
def amortized_maxSW(X, Y,f,detach=True,p=2):
    theta = f(torch.cat([X,Y],dim=0),detach=detach)
    return  one_dimensional_Wasserstein(X,Y,theta,p),theta
def amortized_PRW(X, Y,f,detach=True,p=2,device="cuda"):
    U = f(torch.cat([X,Y],dim=0),detach=detach)
    return  k_dimensional_Wasserstein(X,Y,U,p,device),U
def PRW(X,Y,k,p=2, max_iter=100, lr=1e-2,device="cuda"):
    U = torch.randn(( X.shape[1],k), device=device, requires_grad=True)
    U.data = torch.qr(U)[0]
    opt = torch.optim.Adam([U], lr=lr)
    Xdetach = X.detach()
    Ydetach = Y.detach()
    for _ in range(max_iter):
        l = -k_dimensional_Wasserstein(Xdetach,Ydetach,U,p,device)
        opt.zero_grad()
        l.backward()
        opt.step()
        U.data = torch.qr(U)[0]
    wasserstein_distance=k_dimensional_Wasserstein(X,Y,U,p,device)
    return wasserstein_distance,U