import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import DnCNN, ResNetSNN,resnet18
from dataset_gray import prepare_data, Dataset
from utils import *
from torchsummary import summary

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
net_pth = "net_num7_g3.pth"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=10, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=10, help='noise level used on validation set')
opt = parser.parse_args()

def add_stripe_noise(img, intensity=0.1, stripe_prob=0.5):
    """
    img: (B, C, H, W) normalized image (0~1)
    intensity: max intensity of stripe noise (0~1)
    stripe_prob: 확률적으로 몇 개의 줄에 줄무늬 넣을지 결정
    """
    B, C, H, W = img.shape
    noise = torch.zeros_like(img)
    for b in range(B):
        for h in range(H):
            if torch.rand(1).item() < stripe_prob:
                stripe_value = (torch.rand(1).item() * 2 - 1) * intensity  # [-intensity, intensity]
                # noise[b, :, :, w] = stripe_value  # 동일한 값으로 한 column 전체에 추가 세로노이즈
                noise[b, :, h, :] = stripe_value  # 모든 채널, 모든 width(W)에 대해 같은 행 h 가로노이즈
    return noise

def add_shot_noise(img, noise_strength=0.3):
    img_scaled = img * 255
    noisy = torch.poisson(img_scaled)
    noisy_img = (1 - noise_strength) * img_scaled + noise_strength * noisy
    noisy_img = torch.clamp(noisy_img / 255, 0, 1)
    return noisy_img

def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    # net = DnCNN(channels=1)
    net = resnet18()    
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    # summary(model, input_size=(1,64,64), device='cuda')
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    noiseL_B=[0,55] # ingnored when opt.mode=='S'
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data
            if opt.mode == 'S':
                # noise = add_shot_noise(img_train)
                # noise = add_stripe_noise(img_train)
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
            if opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0,:,:,:].size()
                    noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
            imgn_train = img_train + noise
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())
            out_train = model(imgn_train)
            # print(f"Input: {img_train.shape}, imgn_val: {imgn_train.shape}, noise: {noise.shape}, out_train: {out_train.shape}")
            loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            ssim_train = batch_SSIM(out_train, img_train, data_range=1.0)  # ✅ 추가
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f SSIM_train: %.4f" %
                    (epoch+1, i+1, len(loader_train), loss.item(), psnr_train, ssim_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
                writer.add_scalar('SSIM on training data', ssim_train, step)  # ✅ 추가
            step += 1
        ## the end of each epoch
        model.eval()
        # validate
        import torchvision.transforms as transforms

        resize = transforms.Resize((64, 64))
        psnr_val = 0
        ssim_val = 0  # ✅ 추가
        for k in range(len(dataset_val)):
            img_val = resize(dataset_val[k])  # (C, H, W) -> (1, 64, 64)
            img_val = torch.unsqueeze(img_val, 0)  # 배치 차원 추가 -> (1, 1, 64, 64)
            # print(f"Input:{dataset_val[k].shape}")
            noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
            # noise = add_stripe_noise(img_val)
            # noise = add_shot_noise(img_val)
            noise = noise.to(img_val.device)
            imgn_val = img_val + noise
            img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)            
            out_val_model=model(imgn_val)
            # print(f"Input: {img_val.shape}, imgn_val: {imgn_val.shape}, out_val_model: {out_val_model.shape}")
            out_val = torch.clamp(imgn_val-out_val_model, 0., 1.)
            psnr_val += batch_PSNR(out_val, img_val, 1.)
            ssim_val += batch_SSIM(out_val, img_val, data_range=1.0)  # ✅ 추가
        psnr_val /= len(dataset_val)
        ssim_val /= len(dataset_val)  # ✅ 평균 SSIM 계산
        print("\n[epoch %d] PSNR_val: %.4f  SSIM_val: %.4f" % (epoch+1, psnr_val, ssim_val))  # ✅ 출력 확장
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        writer.add_scalar('SSIM on validation data', ssim_val, epoch)  # ✅ 추가
        # log the images
        out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        torch.save(model.state_dict(), os.path.join(opt.outf, net_pth))

if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=64, stride=16, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
    main()
