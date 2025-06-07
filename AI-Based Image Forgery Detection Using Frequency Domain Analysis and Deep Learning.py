import numpy as np
import cv2
import torch
import torch.nn as nn
import random
from google.colab import files
from io import BytesIO
import matplotlib.pyplot as plt

IMG_SIZE = 128
EPOCHS = 3
BATCH_SIZE = 16

class FFTForgeryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

def fft_transform(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    spectrum = 20 * np.log(np.abs(fshift) + 1)
    spectrum = cv2.resize(spectrum, (IMG_SIZE, IMG_SIZE))
    spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())
    return spectrum.astype(np.float32)

def generate_real_image():
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    for _ in range(3):
        cv2.circle(img, (random.randint(20, 108), random.randint(20, 108)), random.randint(10, 20), (255, 255, 255), -1)
    return img

def generate_forged_image():
    img = generate_real_image()
    x, y, w, h = 20, 20, 40, 40
    region = img[y:y+h, x:x+w].copy()
    img[y+10:y+h+10, x+10:x+w+10] = region
    return img

from torch.utils.data import Dataset, DataLoader

class FFTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        x = torch.tensor(self.images[idx]).unsqueeze(0)
        y = torch.tensor(self.labels[idx])
        return x, y

def train_model():
    print("[INFO] Generating synthetic data for training...")
    images, labels = [], []
    for _ in range(100):
        images.append(fft_transform(generate_real_image()))
        labels.append(0)
        images.append(fft_transform(generate_forged_image()))
        labels.append(1)

    train_loader = DataLoader(FFTDataset(images, labels), batch_size=BATCH_SIZE, shuffle=True)
    model = FFTForgeryCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        total, correct = 0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total += y.size(0)
            correct += (output.argmax(1) == y).sum().item()
        print(f"[Epoch {epoch+1}] Accuracy: {100 * correct / total:.2f}%")

    torch.save(model.state_dict(), "fft_model.pth")
    return model

def load_model():
    model = FFTForgeryCNN()
    try:
        model.load_state_dict(torch.load("fft_model.pth", map_location=torch.device("cpu")))
        print("[INFO] Loaded pretrained model.")
    except:
        print("[INFO] Pretrained model not found, training model now...")
        model = train_model()
    model.eval()
    return model

def upload_image():
    uploaded = files.upload()
    if not uploaded:
        print("❌ No file uploaded.")
        return None, None
    for fn in uploaded.keys():
        print(f"Uploaded file: {fn}")
        file_bytes = uploaded[fn]
        # Convert bytes to numpy array for cv2
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            print("❌ Could not read the image file.")
            return None, None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb, fn
    return None, None

def predict(model, img_rgb):
    fft_img = fft_transform(img_rgb)
    tensor = torch.tensor(fft_img).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output).item()
    label = "Forged" if pred == 1 else "Real"
    return fft_img, label

def main():
    model = load_model()
    img, filename = upload_image()
    if img is None:
        return
    print("[INFO] Predicting...")
    spectrum, result = predict(model, img)
    print(f"\n✅ Prediction for uploaded image '{filename}': {result}")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(spectrum, cmap='gray')
    plt.title("FFT Spectrum")
    plt.axis('off')

    plt.suptitle(f"Prediction: {result}", fontsize=16)
    plt.show()

if __name__ == "__main__":
    main()
