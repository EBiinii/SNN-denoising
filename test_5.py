import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN, ResNetSNN,resnet18
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--logfile", type=str, default="net_sb.pth", help='noise level used on test set')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=10, help='noise level used on test set')

opt = parser.parse_args()

def normalize(data):
    return data/255.

def add_stripe_noise(img, intensity=0.2, stripe_prob=0.5):
    B, C, H, W = img.shape
    noise = torch.zeros_like(img)
    for b in range(B):
        for h in range(H):
            if torch.rand(1).item() < stripe_prob:
                stripe_value = (torch.rand(1).item() * 2 - 1) * intensity
                # noise[b, :, :, w] = stripe_value  # 동일한 값으로 한 column 전체에 추가 세로노이즈
                noise[b, :, h, :] = stripe_value  # 모든 채널, 모든 width(W)에 대해 같은 행 h 가로노이즈
    return noise

def add_shot_noise(img, noise_strength=0.2):
    img_scaled = img * 255
    noisy = torch.poisson(img_scaled)
    noisy_img = (1 - noise_strength) * img_scaled + noise_strength * noisy
    noisy_img = torch.clamp(noisy_img / 255, 0, 1)
    return noisy_img

def main():
    # Build model
    print('Loading model ...\n')
    net = resnet18()
    # net = DnCNN(channels=1)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, opt.logfile)))
    model.eval()

    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()

    # 전체 5번 반복
    psnr_runs = []
    ssim_runs = []
    for repeat in range(5):
        psnr_test = 0
        ssim_test = 0
        for f in files_source:
            # image load
            Img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            Img = cv2.resize(Img, (64, 64))
            Img = normalize(np.float32(Img))
            Img = np.expand_dims(Img, 0)
            Img = np.expand_dims(Img, 1)
            ISource = torch.Tensor(Img)

            # noise
            # noise = add_stripe_noise(ISource)
            # noise = add_shot_noise(ISource)
            noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)

            INoisy = ISource + noise
            ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())

            with torch.no_grad():
                Out = torch.clamp(INoisy-model(INoisy), 0., 1.)

            psnr = batch_PSNR(Out, ISource, 1.)
            ssim_val = batch_SSIM(Out, ISource, data_range=1.0)

            psnr_test += psnr
            ssim_test += ssim_val

        psnr_test /= len(files_source)
        ssim_test /= len(files_source)

        psnr_runs.append(psnr_test)
        ssim_runs.append(ssim_test)

        print(f"[Run {repeat+1}] PSNR: {psnr_test:.4f}, SSIM: {ssim_test:.4f}")

    # 5회 평균
    print("\n========== Final Result (5 runs average) ==========")
    print(f"Mean PSNR: {np.mean(psnr_runs):.4f} (std {np.std(psnr_runs):.4f})")
    print(f"Mean SSIM: {np.mean(ssim_runs):.4f} (std {np.std(ssim_runs):.4f})")

if __name__ == "__main__":
    main()