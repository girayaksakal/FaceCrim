import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import builtins

# Model eğitimi sırasında geçen zamanı görmek için print fonksiyonu
original_print = builtins.print
def print_with_timestamp(*args, **kwargs):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    original_print(f"[{timestamp}]", *args, **kwargs)

builtins.print = print_with_timestamp

# GPU varsa daha hızlı eğitmek için GPU kullan
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Dataset yapısı
# data/
#   ├── örgüt_1/
#   │   ├── isim_SOYİSİM.jpg
#   ├── örgüt_2/
#   │   ├── isim_SOYİSİM.jpg
dataset = datasets.ImageFolder("data_processed/", transform = transform)

# Eğitim, doğrulama ve test verilerini ayırma ve klasörlere yerleştirme
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of testing samples: {len(test_dataset)}")

# Modelin Hazırlanması
model = models.resnet18(weights = None)  # ResNet-18 modelini kullanıyoruz
num_classes = len(dataset.classes) # Toplam örgüt sayısı (klasör sayısı)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss ve optimizasyon fonksiyonları
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Modelin eğitilmesi
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0

    # Eğitim
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # İleri besleme
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Geri yayılım ve optimizasyon
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Doğrulama
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Doğruluk hesaplama
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader):.4f}, "
          f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    
# Test Set ile değerlendirme
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Modeli kaydet
torch.save(model.state_dict(), "test_resnet18_terror_model.pth")
print("Model saved as 'test_resnet18_terror_model.pth'")