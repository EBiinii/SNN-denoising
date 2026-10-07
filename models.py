import torch
import torch.nn as nn
import torch.nn.functional as F
# imports
import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

# neuron and simulation parameters
spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5
num_steps = 7

class InhibitoryNeuron(nn.Module):
    def __init__(self, inhibition_strength):
        super(InhibitoryNeuron, self).__init__()
        self.inhibition_strength = inhibition_strength
        
    def forward(self, spk):
        # 스파이크를 억제하는 간단한 메커니즘
        inhibited_spk = spk * (1 - self.inhibition_strength)
        return inhibited_spk

class DnCNN(nn.Module):
    def __init__(self, channels=1, beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25)):
        super().__init__()
        kernel_size = 3
        padding = 1
        features = 64

        # -------------------
        # 1️⃣ 초반 ANN (6 Conv)
        # -------------------
        early_layers = [
            nn.Conv2d(channels, features, kernel_size, padding=padding, bias=False),
            nn.ReLU(inplace=True)
        ]
        for _ in range(5):  # Conv+BN+ReLU 반복 = 5 Conv
            early_layers += [
                nn.Conv2d(features, features, kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ]
        self.early = nn.Sequential(*early_layers)  # 총 6 Conv

        # -------------------
        # 2️⃣ 중간 SNN 블록 
        # -------------------
        self.mid_layers = nn.ModuleList([
            nn.Conv2d(features, features, kernel_size, padding=padding, bias=False),  # 7
            snn.Leaky(beta=beta, spike_grad=spike_grad),
            # InhibitoryNeuron(inhibition_strength=0.5),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),

            nn.Conv2d(features, features, kernel_size, padding=padding, bias=False),  # 8
            snn.Leaky(beta=beta, spike_grad=spike_grad),
            # InhibitoryNeuron(inhibition_strength=0.5),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),

            nn.Conv2d(features, features, kernel_size, padding=padding, bias=False),  # 9
            snn.Leaky(beta=beta, spike_grad=spike_grad),
            # InhibitoryNeuron(inhibition_strength=0.5),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),

            nn.Conv2d(features, features, kernel_size, padding=padding, bias=False),  # 10
            snn.Leaky(beta=beta, spike_grad=spike_grad),
            # InhibitoryNeuron(inhibition_strength=0.5),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        ])
        # Conv 수 = 4 (중간 블록 포함)

        # -------------------
        # 3️⃣ 후반 ANN (6 Conv)
        # -------------------
        late_layers = []
        for _ in range(6):  # Conv+BN+ReLU 반복 = 6 Conv
            late_layers += [
                nn.Conv2d(features, features, kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ]
        self.late = nn.Sequential(*late_layers)

        # -------------------
        # 마지막 Conv (출력)
        # -------------------
        self.final_conv = nn.Conv2d(features, channels, kernel_size, padding=padding, bias=False)

    def forward(self, x, num_steps=num_steps):
        # 1️⃣ 초반 ANN 처리
        out = self.early(x)

        # 2️⃣ 중간 SNN 처리 (timestep 반복)
        mems = [layer.init_leaky() for layer in self.mid_layers if isinstance(layer, snn.Leaky)]
        for t in range(num_steps):
            mem_idx = 0
            for layer in self.mid_layers:
                if isinstance(layer, snn.Leaky):
                    out, mems[mem_idx] = layer(out, mems[mem_idx])
                    mem_idx += 1
                else:
                    out = layer(out)

        # 3️⃣ 후반 ANN 처리
        out = self.late(out)

        # 4️⃣ 마지막 Conv 출력
        out = self.final_conv(out)
        return out

    
class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.residual_function(x) + self.shortcut(x))

class ResNetSNN(nn.Module):
    def __init__(self, block, num_block, num_classes=100):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.inhib1 = InhibitoryNeuron(inhibition_strength=0.5)

        # Decoder: 출력 크기 8x8 → 64x64 upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512 * block.expansion, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=3, padding=1),  # 최종 출력 채널 1 (복원된 이미지)
            # 활성화 함수는 loss 계산 전에 clamp 등으로 처리
        )



    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        
        batch_size = x.size(0)  # 입력 배치 크기
        mem_fc = torch.zeros(batch_size, 512, 8, 8).cuda()

        # spike_inp = poisson_gen(x)

        cur1 = self.conv1(x)
        # cur1 = self.att1(cur1)
       

        for t in range(num_steps):
          cur2 = self.conv2_x(cur1)
          spk2, mem2 = self.lif1(cur2, mem2) 
        #   spk2 = self.inhib1(spk2)             
          spk2 = spk2 + cur2 # residual connection

          cur3 = self.conv3_x(spk2)
          spk3, mem3 = self.lif2(cur3, mem3)
        #   spk3 = self.inhib1(spk3)   
          spk3 = spk3 + cur3 # residual connection

          cur4 = self.conv4_x(spk3)
          spk4, mem4 = self.lif3(cur4, mem4)
          spk4 = self.inhib1(spk4)   
          spk4 = cur4 + spk4 # residual connection

          cur5 = self.conv5_x(spk4)
          spk5, mem5 = self.lif4(cur5, mem5)
        #   spk5 = self.inhib1(spk5)   
          spk5 = spk5 + cur5 # residual connection

          spk5 = F.adaptive_avg_pool2d(spk5, (8, 8))  # 여기 추가
          mem_fc = mem_fc + spk5
        
        out_enc = mem_fc / num_steps  # 평균화
        
        out_dec = self.decoder(out_enc)  # upsampling → 출력 크기 (B,1,64,64)
        # # out_voltage = self.att4(out_voltage)
        # output = self.avg_pool(out_voltage)
        # output = output.view(output.size(0), -1)
        # output = self.fc(output)

        return out_dec

def resnet18(num_classes=10, **kargs):
    """ return a ResNet 18 object
    """
    return ResNetSNN(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def resnet34(num_classes=10, **kargs):
    """ return a ResNet 34 object
    """
    return ResNetSNN(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet50(num_classes=10, **kargs):
    """ return a ResNet 50 object
    """
    return ResNetSNN(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)
