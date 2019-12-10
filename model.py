import torch
import torch.nn as nn
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

class convnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,64,5, stride = 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride=1), # 5
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear( 12 * 12 * 64, 2048), # 12 12
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 50),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(1,64,3, stride = 1), # channel depth 64
            nn.ReLU(),
            nn.Conv2d(64, 64,3, stride=1), # channel depth 64
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear( 14 * 14 * 64, 3136), # channel depth 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(3136, 50),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc3 = nn.Sequential(
            nn.Linear( 100 , 50),
        )


        # 양날개 전 버전
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

    def forward(self, x):
        x_1 = x.clone()

        x_1 = self.layer1(x_1)
        x_1 = x_1.view(x_1.shape[0], -1)
        x_1 = self.fc1(x_1)

        x = self.layer2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc2(x)

        x = torch.cat([x, x_1], dim=1)
        x = self.fc3(x)
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
        return x

    #######################################################
    #######################################################
    #######################################################
    #######################################################
    #현성 양날개
# class convnet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1, 64, 5, stride=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 5, stride=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.fc1 = nn.Sequential(
#             nn.Linear(12 * 12 * 64, 2048),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.fc2 = nn.Linear(2048, 50)
#
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, stride=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, 3, stride=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.fc3 = nn.Linear(4 * 4 * 128, 50)
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x_1 = x.clone()
#
#         x = x.view(-1, 12 * 12 * 64)
#         x = self.fc1(x)
#         x = self.fc2(x)
#
#         x_1 = self.layer2(x_1)
#         x_1 = x_1.view(-1, 4 * 4 * 128)
#         x_1 = self.fc3(x_1)
#
#         return x + x_1