import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from PIL import Image

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modeli ve sınıfları yükle
def load_model(model_path, device):
    model = models.resnet18(weights=None) 
    num_classes = 3
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, weights_only= True))
    model = model.to(device)
    model.eval()
    return model

# Tahmin fonksiyonu
def predict_image(image_path, model, class_names, device):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    return class_names[predicted.item()]

# Modeli yükle
model_path = "new_resnet18_terror_model.pth"
model = load_model(model_path, device)

# Sınıf isimlerini modelden alarak yükle
dataset = datasets.ImageFolder("data_processed/")
class_names = dataset.classes


# Test edilecek görselin yolu
image_path = "data_samples_processed/innocent_sample/000027.jpg"

# Tahmin yap
predicted_class = predict_image(image_path, model, class_names, device)
print(f"Predicted Class: {predicted_class}")