import torch
import torch.nn as nn
import torch.nn.functional as F
#12/06
# normalization -1 ~ 1
# edge cv2
# batch size > 1
# kernel size 5 % padding 2 > 3404 / 5000
# fc size > 3265 / 5000
# epoch
# output channel 더 깊게
# layer 상범 버전으로 (stride =2)dropout 제거 > acc : 2946 / 5000
# layer 상범 버전으로 (stride =1)dropout 제거 > 4140 / 5000
# reshape 이상한 4191 / 5000
# 위와 동일 2 epoch 4583 / 5000


#######################################################
#######################################################
#######################################################
#######################################################
# 김양곤 양날개 버전
# class convnet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1,64,5, stride = 1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 5, stride=1), # 5
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.fc1 = nn.Sequential(
#             nn.Linear( 12 * 12 * 64, 2048), # 12 12
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(2048, 512),
#             nn.LeakyReLU(inplace=True), # leaky relu default 0.01일때도 잘 됐음. inplace 넣으니깐 좀 빨라지긴 함
#         )
#
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(1,64,3, stride = 1), # channel depth 64
#             nn.ReLU(),
#             nn.Conv2d(64, 64,3, stride=1), # channel depth 64
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear( 14 * 14 * 64, 2048), # channel depth 64
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(2048, 784),
#             nn.LeakyReLU(inplace=True),
#         )
#         self.fc3 = nn.Sequential(
#             nn.Linear(1296, 50),
#         )
#
#     def forward(self, x):
#         x_1 = x.clone()
#
#         x_1 = self.layer1(x_1)
#         x_1 = x_1.view(x_1.shape[0], -1)
#         x_1 = self.fc1(x_1)
#
#         x = self.layer2(x)
#         x = x.view(x.shape[0], -1)
#         x = self.fc2(x)
#
#         x = torch.cat([x, x_1], dim=1)
#         x = self.fc3(x)
#         return x

        #######################################################
        #######################################################
        #######################################################
        #######################################################
        # 양날개 전 버전
# class convnet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Sequential(
#             # nn.Conv2d(1, 6, 5, stride = 1, padding = 2),
#             nn.Conv2d(1,64,5, stride = 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             # nn.MaxPool2d(2)
#         )
#         self.layer2 = nn.Sequential(
#             # nn.Conv2d(6, 16, 5, stride = 1, padding = 2),
#             nn.Conv2d(64, 64, 3, stride = 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
# #         self.layer3 = nn.Sequential(
# #             nn.Conv2d(32, 64, 3, stride = 1),
# #             nn.BatchNorm2d(32),
# #             nn.ReLU(),
# #             nn.MaxPool2d(2)
# #         )
#         self.layer4 = nn.Sequential(
#             nn.Linear( 13 * 13 * 64, 2048),
#             nn.LeakyReLU(),
#         )
# #         self.layer5 = nn.Sequential(
# #             nn.Linear(150, 100),
# #             nn.ReLU()
# #         )
#         self.layer6 = nn.Sequential(
#             # nn.Dropout(0.3),
#             nn.Linear(2048, 50),
#             nn.LeakyReLU(),
#         )
#
#         self.layer7 = nn.Sequential(
#             nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
#             nn.Conv2d(1, 32, 5, stride=1, padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, 3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.layer8 = nn.Sequential(
#             nn.Linear(512, 50),
#             nn.LeakyReLU(),
#         )
#         self.layer9 = nn.Sequential(
#             nn.Linear(100, 50)
#         )

    # def forward(self, x):
#         x_1 = x.clone()
#         x_1 = self.layer7(x_1)
#         x_1 = x_1.view(x_1.shape[0], -1)
#         x_1 = self.layer8(x_1)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
# #         x = self.layer3(x)
#         x = x.view(-1, 13* 13 * 64)
#         x = self.layer4(x)
# #         x = self.layer5(x)
#         x = self.layer6(x)
#
#         x=torch.cat([x,x_1],dim=1)
#         x = self.layer9(x)
#         return x

    #######################################################
    #######################################################
    #######################################################
    #######################################################
    # 현성 양날개 를 내 맛대로 변화
    # 이거 불리합니적 100 50 concat 4분 40초 97.56 filter를 3 3 > 3 3으로 줄이게 되면 더 좋다.
    # 이거 합리적 1296 50 96.22 4분 30초

    # concat 하지말고, x+x_1 해야한다.

# class convnet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1, 64, 3, stride=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, stride=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.fc1 = nn.Sequential(
#             nn.Linear(14 * 14 * 64, 3136),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(3136, 50),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#         # self.fc2 = nn.Linear(2048, 50)
#
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, stride=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, 3, stride=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.fc3 = nn.Sequential(
#             nn.Linear(5 * 5 * 128, 50),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#
#         # self.fc4 = nn.Linear(100, 50)
#     def forward(self, x):
#         x = self.layer1(x)
#         x_1 = x.clone()
#
#         x = x.view(x.shape[0], -1)
#         x = self.fc1(x)
#         # x = self.fc2(x)
#
#         x_1 = self.layer2(x_1)
#         x_1 = x_1.view(x_1.shape[0], -1)
#         x_1 = self.fc3(x_1)
#
#         # x = torch.cat([x,x_1],dim=1)
#         # x = self.fc4(x)
#         return x+x_1


#########################
#########################
#########################
#########################3단
# class convnet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1, 16, 5, stride=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 16, 5, stride=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.fc1 = nn.Sequential(
#             nn.Linear(12 * 12 * 16, 576), #
#             nn.LeakyReLU(0.2, inplace=True)
#             # nn.Linear(576, 114), #
#             # nn.LeakyReLU(0.2, inplace=True)
#         )
#
#         # 12 by 12 들어간다.
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(16, 64, 3, stride=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, stride=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(4 * 4 * 64, 256),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#
#         # 4 by 4 들어간다.
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, stride=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         # self.fc3 = nn.Sequential(
#         #     nn.Linear(1 * 1 * 128, 128),
#         #     nn.LeakyReLU(0.2, inplace=True)
#         # )
#
#         self.concatfc = nn.Linear(960, 50)
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x_1 = x.clone()
#
#         x = x.view(x.shape[0], -1)
#         x = self.fc1(x)   # 576
#
#
#         x_1 = self.layer2(x_1)
#         x_2 = x_1.clone()
#
#         x_1 = x_1.view(x_1.shape[0], -1)
#         x_1 = self.fc2(x_1) # 256
#
#         x_2 = self.layer3(x_2)
#         x_2 = x_2.view(x_2.shape[0], -1) # 128
#
#         x = torch.cat([x,x_1,x_2],dim=1)
#         x = self.concatfc(x)
#         return x



# 김양곤 양날개 버전 진화
# 4분 37초
# 96.66
#
# class convnet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1,64,7, stride = 1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 5, stride=1), # 5
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.fc1 = nn.Sequential(
#             nn.Linear( 11 * 11 * 64, 2048), # 12 12
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(2048, 512),
#             nn.LeakyReLU(inplace=True), # leaky relu default 0.01일때도 잘 됐음. inplace 넣으니깐 좀 빨라지긴 함
#         )
#
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(1,64,3, stride = 1), # channel depth 64
#             nn.ReLU(),
#             nn.Conv2d(64, 64,3, stride=1), # channel depth 64
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear( 14 * 14 * 64, 2048), # channel depth 64
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(2048, 784),
#             nn.LeakyReLU(inplace=True),
#         )
#         self.fc3 = nn.Sequential(
#             nn.Linear(1296, 50),
#         )
#
#     def forward(self, x):
#         x_1 = x.clone()
#
#         x_1 = self.layer1(x_1)
#         x_1 = x_1.view(x_1.shape[0], -1)
#         x_1 = self.fc1(x_1)
#
#         x = self.layer2(x)
#         x = x.view(x.shape[0], -1)
#         x = self.fc2(x)
#
#         x = torch.cat([x, x_1], dim=1)
#         x = self.fc3(x)
#         return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 이거 반날개(현성+양곤)인데, 3 3 3 3 이고 concat 안하고 x+x_1인데 p100에서 97.38% 4분 20초이다. / 같은 실험 vm으로 땡겨와서 3분 37초 97.46
# 일단 leaky leru 를 0.001로 96.68%
class convnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.swish = Swish()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(13 * 13 * 64, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 50),
            # nn.LeakyReLU(0.2, inplace=True)
        )
        # self.fc2 = nn.Linear(2048, 50)

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(1 * 1 * 128, 50),
            # nn.LeakyReLU(0.2, inplace=True)
        )

        # self.fc4 = nn.Linear(100, 50)
    def forward(self, x):
        x = self.layer1(x)
        x_1 = x.clone()

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        # x = self.fc2(x)

        x_1 = self.layer2(x_1)
        x_1 = x_1.view(x_1.shape[0], -1)
        x_1 = self.fc3(x_1)

        # x = torch.cat([x,x_1],dim=1)
        # x = self.fc4(x)
        return x+x_1