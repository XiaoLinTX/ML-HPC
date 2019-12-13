import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt

# Hyper Parameters
EPOCH = 1  # 训练整批数据多少次
BATCH_SIZE = 50
LR = 0.0002  # 学习率
DOWNLOAD_MNIST = True  # 已经下载好的话，会自动跳过的
len_Z = 100  # random input.channal for Generator
g_hidden_channal = 64
d_hidden_channal = 64
image_channal = 1  # mnist数据为黑白的只有一维

# Mnist 手写数字
train_data = torchvision.datasets.MNIST(
    root='./mnist/',  # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,  # 没下载就下载, 下载了就不用再下了
)

test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False
)

# 训练集丢BATCH_SIZE个, 图片大小为28*28
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True  # 是否打乱顺序
)


class Generator(nn.Module):
    def __init__(self, len_Z, hidden_channal, output_channal):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=len_Z,
                out_channels=hidden_channal * 4,
                kernel_size=4,
            ),
            nn.BatchNorm2d(hidden_channal * 4),
            nn.ReLU()
        )
        # [BATCH, hidden_channal * 4 , 4, 4]
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hidden_channal * 4,
                out_channels=hidden_channal * 2,
                kernel_size=3,  # 保证生成图像大小为28
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(hidden_channal * 2),
            nn.ReLU()
        )
        #
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hidden_channal * 2,
                out_channels=hidden_channal,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(hidden_channal),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hidden_channal,
                out_channels=output_channal,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Tanh()
        )

    def forward(self, x):
        # [50, 100, 1, 1]
        out = self.layer1(x)
        # [50, 256, 4, 4]
        # print(out.shape)
        out = self.layer2(out)
        # [50, 128, 7, 7]
        # print(out.shape)
        out = self.layer3(out)
        # [50, 64, 14, 14]
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        # [50, 1, 28, 28]
        return out


# # Test Generator
# G = Generator(len_Z, g_hidden_channal, image_channal)
# data = torch.randn((BATCH_SIZE, len_Z, 1, 1))
# print(G(data).shape)


class Discriminator(nn.Module):
    def __init__(self, input_channal, hidden_channal):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channal,
                out_channels=hidden_channal,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(hidden_channal),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channal,
                out_channels=hidden_channal * 2,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(hidden_channal * 2),
            nn.LeakyReLU(0.2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channal * 2,
                out_channels=hidden_channal * 4,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(hidden_channal * 4),
            nn.LeakyReLU(0.2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channal * 4,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0
            ),
            nn.Sigmoid()
        )
        # [BATCH, 1, 1, 1]

    def forward(self, x):
        # print(x.shape)
        out = self.layer1(x)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)

        return out


# # Test Discriminator
# D = Discriminator(1, d_hidden_channal)
# data = torch.randn((BATCH_SIZE, 1, 28, 28))
# print(D(data).shape)


G = Generator(len_Z, g_hidden_channal, image_channal)
D = Discriminator(image_channal, g_hidden_channal)

# loss & optimizer
criterion = nn.BCELoss()
optimD = torch.optim.Adam(D.parameters(), lr=LR)
optimG = torch.optim.Adam(G.parameters(), lr=LR)

label_Real = torch.FloatTensor(BATCH_SIZE).data.fill_(1)
label_Fake = torch.FloatTensor(BATCH_SIZE).data.fill_(0)

for epoch in range(EPOCH):
    for step, (images, imagesLabel) in enumerate(train_loader):
        G_ideas = torch.randn((BATCH_SIZE, len_Z, 1, 1))

        G_paintings = G(G_ideas)
        prob_artist0 = D(images)  # D try to increase this prob
        prob_artist1 = D(G_paintings)
        p0 = torch.squeeze(prob_artist0)
        p1 = torch.squeeze(prob_artist1)

        errD_real = criterion(p0, label_Real)


        errD_fake = criterion(p1, label_Fake)
        # errD_fake.backward()

        errD = errD_fake + errD_real
        errG = criterion(p1, label_Real)

        optimD.zero_grad()
        errD.backward(retain_graph=True)
        optimD.step()

        optimG.zero_grad()
        errG.backward(retain_graph=True)
        optimG.step()
        if (step+1) % 100 == 0:
            picture = torch.squeeze(G_paintings[0]).detach().numpy()
            plt.imshow(picture, cmap=plt.cm.gray_r)
            plt.show()

