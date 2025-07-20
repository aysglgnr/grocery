import pandas as pd

# Dosya yolu
train_file = "GroceryStoreDataset/dataset/train.txt"

# Veriyi okuma ve virgülleri temizleme
df_train = pd.read_csv(
    train_file,
    sep=' ',
    header=None,
    names=['image_path', 'fine_label', 'coarse_label'],
    skipinitialspace=True,
    converters={'fine_label': lambda x: int(x.replace(',', '')),
                'coarse_label': lambda x: int(x.replace(',', ''))}
)

# Görsel yollarındaki virgülleri temizleme
df_train['image_path'] = df_train['image_path'].apply(lambda x: x.replace(',', ''))

print(df_train.head())
print(df_train.dtypes)


import matplotlib.pyplot as plt
from PIL import Image
import os

# Görsellerin bulunduğu ana dizin
base_image_path = r"GroceryStoreDataset/dataset"


# # İlk 5 görseli çizme
# for i in range(5):
#     row = df_train.iloc[i]
#     img_path = os.path.join(base_image_path, row['image_path'])

#     try:
#         img = Image.open(img_path)
#         plt.imshow(img)
#         plt.title(f"Fine: {row['fine_label']} - Coarse: {row['coarse_label']}")
#         plt.axis('off')
#         plt.show()
#     except Exception as e:
#         print(f"❌ Hata: {img_path} yüklenemedi. {e}")


import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models #pip install torchvision
import torch #pip install torch

class GroceryDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.iloc[idx]['image_path'])
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx]['fine_label']  # veya 'coarse_label' da olabilir

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet ortalaması
                         [0.229, 0.224, 0.225])
])


# Dosya yolu ayarları
base_path = r"GroceryStoreDataset/dataset"
train_file = os.path.join(base_path, "train.txt")

# Veriyi okuma
df_train = pd.read_csv(
    train_file,
    sep=' ',
    header=None,
    names=['image_path', 'fine_label', 'coarse_label'],
    skipinitialspace=True,
    converters={
        'fine_label': lambda x: int(x.replace(',', '')),
        'coarse_label': lambda x: int(x.replace(',', ''))
    }
)
df_train['image_path'] = df_train['image_path'].apply(lambda x: x.replace(',', ''))

# Dataset ve DataLoader oluşturma
train_dataset = GroceryDataset(df_train, base_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


images, labels = next(iter(train_loader))
print(f"Görüntü boyutu: {images.shape}")
print(f"Etiketler: {labels[:5]}")

import torch.nn as nn
from torchvision import models

# Cihaz seçimi (GPU varsa kullanma)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pretrained ResNet18 modeli
model = models.resnet18(pretrained=True)

# Son katmanı 81 sınıfa göre güncelleme
num_classes = df_train['fine_label'].nunique()
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Sıfırlama
        optimizer.zero_grad()

        # Tahmin
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Kayıp ve accuracy hesaplama
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")




