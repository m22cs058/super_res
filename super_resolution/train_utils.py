from tqdm import tqdm
import os
import torch
from torch.autograd import Variable

def train(trainloader, N_EPOCHS, netG, netD, optimizerG, optimizerD, generator_criterion, device):
    results = {"d_loss":[], "g_loss":[]}
    for epoch in range(1, N_EPOCHS + 1):
        train_bar = tqdm(trainloader)
        running_results = {'batch_sizes':0, 'd_loss':0,
                            "g_loss":0, "d_score":0, "g_score":0}

        netG.train()
        netD.train()
        for data, target in train_bar:
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            real_img = Variable(target)
            real_img = real_img.to(device)
            z = Variable(data)
            z = z.to(device)

            ## Update Discriminator ##
            fake_img = netG(z)
            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph = True)
            optimizerD.step()
            
            ## Now update Generator
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            netG.zero_grad()
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()

            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerG.step()

            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size

            ## Updating the progress bar
            train_bar.set_description(desc="[%d/%d] Loss_D: %.4f Loss_G: %.4f" % (
                epoch, N_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes']
            ))
            if epoch % 10 == 0:
                torch.save(netG.state_dict(), 'models/netG3_epoch{}.pt'.format(epoch))
    
        results['g_loss'].append(running_results['d_loss'] / len(trainloader.dataset))
        results['d_loss'].append(running_results['g_loss'] / len(trainloader.dataset))

    return netG, netD, results['g_loss'], results['d_loss']