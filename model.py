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
            # nn.Conv2d(1, 6, 5, stride = 1, padding = 2),
            nn.Conv2d(1,64,7, stride = 1),
#             nn.BatchNorm2d(16),
            nn.ReLU(),
#             nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            # nn.Conv2d(6, 16, 5, stride = 1, padding = 2),
            nn.Conv2d(64, 64, 3, stride = 1),
#             nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(32, 64, 3, stride = 1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
        self.layer4 = nn.Sequential(
            nn.Linear( 12 * 12 * 64, 2048),
            nn.ReLU()
        )
#         self.layer5 = nn.Sequential(
#             nn.Linear(150, 100),
#             nn.ReLU()
#         )
        self.layer6 = nn.Sequential(
            # nn.Dropout(0.3),
            nn.Linear(2048, 50)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
#         x = self.layer3(x)
        x = x.view(-1, 12* 12 * 64)
        x = self.layer4(x)
#         x = self.layer5(x)
        return self.layer6(x)