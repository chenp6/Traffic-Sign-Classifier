import torch
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import gc

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix


import random
import numpy as np

import time

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



# Set seed = 0 for random, numpy, and torch
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




# pickle載入
import pickle
training_file = "traffic-signs-data/train.p"
validation_file = "traffic-signs-data/valid.p"
testing_file = "traffic-signs-data/test.p"
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

x_train, y_train = train['features'], train['labels']
x_valid, y_valid = valid['features'], valid['labels']
x_test, y_test = test['features'], test['labels']

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_valid shape:", x_valid.shape)
print("y_valid shape:", y_valid.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# 定義資料集
class TrafficSignsDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.classes = np.unique(labels)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# 定義轉換
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



# batch大小
batch_size = 64



"""
資料集設定
"""
# train dataset path 
# 建立train資料集並transform格式
train_dataset = TrafficSignsDataset(x_train, y_train, transform=transform)
# 載入dataset並拆分batch
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# validation dataset path 
# 建立valid資料集並transform格式
valid_dataset = TrafficSignsDataset(x_valid, y_valid, transform=transform)
# 載入dataset並拆分batch
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# test dataset path 
# 建立test資料集並transform格式
test_dataset = TrafficSignsDataset(x_test, y_test, transform=transform)
# 載入dataset並拆分batch
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(np.unique(y_train))
print(f'Number of unique classes: {num_classes}')

"""
模型輸出檔案
"""
trained_model_name = "Alexnet_v1_baseline.pth"

"""
模型架構 : v1為Alexnet baseline
"""
# 以AlexNet為baseline進行的CustomAlexNetModel
class CustomAlexNetModel(nn.Module):
    def __init__(self):
        super(CustomAlexNetModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),


            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # nn.Conv2d(192, 384, kernel_size=3, padding=1), # 輸出維度調整 
            # nn.ReLU(inplace=True),
            # nn.Conv2d(384, 256, kernel_size=3, padding=1), # 移除此層
            # nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 43),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



# 檢查是否有可用的GPU，如果有就將device設置為cuda，否則設置為cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ## add
print(torch.cuda.is_available()) #是否有可用的CUDA加速

# 架構確認，查看參數設定及大小
import torchsummary
torchsummary.summary(CustomAlexNetModel().to(device), (3, 224,224), batch_size=batch_size)



# 釋放記憶體，避免記憶體已滿導致訓練失敗
torch.cuda.empty_cache()
gc.collect()


"""
模型訓練
"""
epochs = 60  # 訓練epoch次數
training_loss = []
validation_loss = []
training_accuracy = []
validation_accuracy = []
def train_model():       
    model = CustomAlexNetModel() # 使用CustomAlexNetModel模型架構
    model = model.to(device) 

    learning_rate = 0.001  # 初始學習率

    # 建立 Adam Optimizer (優化調整learing rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

    
    # 使用交叉熵損失作為loss function
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        # 切換至訓練模式
        model.train()

        # 暫存計數器設定與歸零
        running_loss = 0.0 # loss暫存
        running_correct = 0 # 正確個數暫存
        running_total = 0 # 總數暫存

        # batch 訓練循環
        for batch in train_loader:
            
            # 提取 batch 的 資料（inputs 和 labels）
            inputs, labels = batch
            # 將資料（inputs 和 labels）移動到指定的設備上，以加速運算
            inputs, labels = inputs.to(device), labels.to(device)
            # 梯度歸零
            optimizer.zero_grad()
            # 預測 : 向前傳播
            outputs = model(inputs)
            # 計算loss : 交叉熵
            loss = criterion(outputs, labels)  

            # 反向傳播
            loss.backward()

            # 更新模型的參數
            optimizer.step()

            # 結果儲存
            _, predicted = torch.max(outputs.data, 1) # 預測結果
            running_correct += (predicted == labels).sum().item() # 預測正確者加入
            running_total += labels.size(0) # 更新epoch的training資料累計筆數 (根據此batch大小)
            running_loss += loss.item() #更新目前的epoch train loss sum

        # 計算並儲存epoch training 平均 loss
        training_loss.append(running_loss / running_total) 

        # 計算並儲存 epoch training 正確率
        training_accuracy.append(100 * running_correct / running_total)


        # 切換至驗證模式
        model.eval()

        # 暫存計數器設定與歸零
        running_loss = 0.0 # loss暫存
        running_correct = 0 # 正確個數暫存
        running_total = 0 # 總數暫存

        with torch.no_grad(): # 禁用梯度運算(不需要進行反向傳播)

            # batch 驗證循環
            for batch in valid_loader:

                # 提取 batch 的 資料（inputs 和 labels）
                inputs, labels = batch
                # 將資料（inputs 和 labels）移動到指定的設備上，以加速運算
                inputs, labels = inputs.to(device), labels.to(device)
                # 預測 : 向前傳播
                outputs = model(inputs)                                        
                # 計算loss : 交叉熵
                loss = criterion(outputs, labels)  


                # 結果儲存
                _, predicted = torch.max(outputs.data, 1) # 預測結果
                running_correct += (predicted == labels).sum().item() # 預測正確者加入
                running_total += labels.size(0) # 更新epoch的validation資料累計筆數 (根據此batch大小)
                running_loss += loss.item()  #更新目前的epoch valid loss sum
            
            # 計算並儲存 epoch valid 平均 loss
            validation_loss.append(running_loss / running_total)  

            # 計算並儲存 epoch training 正確率
            validation_accuracy.append(100 * running_correct / running_total)
        
        # print 此epoch訓練/驗證結果
        print(f'Epoch [{epoch}/{epochs-1}] - '
            f'Training Loss: {training_loss[-1]:.4f}, '
            f'Training Accuracy: {training_accuracy[-1]:.2f}%, '
            f'Validation Loss: {validation_loss[-1]:.4f}, '
            f'Validation Accuracy: {validation_accuracy[-1]:.2f}%') 


    # 儲存Model
    torch.save(model.state_dict(),"./model/"+trained_model_name)    



    # Test 階段
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    inference_time = 0

    # 混淆矩陣陣列(預測結果紀錄與正確label)
    all_predictions = []
    all_labels = []

    for batch in test_loader:

        # 提取 batch 的 資料（inputs 和 labels）
        inputs, labels = batch
        # 將資料（inputs 和 labels）移動到指定的設備上，以加速運算
        inputs, labels = inputs.to(device), labels.to(device)

        start_time = time.time()
        # 預測 : 向前傳播
        outputs = model(inputs)                                
        end_time = time.time()
        
        # 計算loss : 交叉熵
        loss = criterion(outputs, labels)  


        # 結果儲存
        _, predicted = torch.max(outputs.data, 1) # 預測結果
        running_correct += (predicted == labels).sum().item() # 預測正確者加入
        running_total += labels.size(0) # 更新epoch的validation資料累計筆數 (根據此batch大小)
        running_loss += loss.item()  #更新目前的epoch valid loss sum

        # extend 類別分類結果 至 混淆矩陣陣列
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 計算並顯示推論時間
        inference_time += end_time - start_time
    

    print(f'Single image inference time: {inference_time/running_total:.6f} seconds')


    # print 測試結果
    print("Test loss:",running_loss / running_total) # 計算平均loss
    print("Test accuracy:",100 * running_correct / running_total) # 計算正確率
    
    
    # 建立混淆矩陣
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    # 繪製混淆矩陣
    classes = [
        "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)",
        "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)", "Speed limit (100km/h)",
        "Speed limit (120km/h)", "No passing", "No passing for vehicles over 3.5 metric tons",
        "Right-of-way at the next intersection", "Priority road", "Yield", "Stop", "No vehicles",
        "Vehicles over 3.5 metric tons prohibited", "No entry", "General caution", "Dangerous curve to the left",
        "Dangerous curve to the right", "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
        "Road work", "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing", "Beware of ice/snow",
        "Wild animals crossing", "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
        "Ahead only", "Go straight or right", "Go straight or left", "Keep right", "Keep left",
        "Roundabout mandatory", "End of no passing", "End of no passing by vehicles over 3.5 metric tons"
    ]

    import seaborn as sns

    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix,cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.show()


    # 繪製各epoch的Accuaracy/Loss圖片
    # 繪製Loss曲線
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.legend()
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 繪製Accuracy曲線
    plt.subplot(1, 2, 2)
    plt.plot(training_accuracy, label='Training Accuracy')
    plt.plot(validation_accuracy, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 100)  # 將 y 軸範圍設置為 0 到 100
    plt.show()

train_model()