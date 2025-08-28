import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder1 = self.conv_block(3, 64, kernel_size=4, stride=2, padding=1, batch_norm=False)
        self.encoder2 = self.conv_block(64, 128, kernel_size=4, stride=2, padding=1)
        self.encoder3 = self.conv_block(128, 256, kernel_size=4, stride=2, padding=1)
        self.encoder4 = self.conv_block(256, 512, kernel_size=4, stride=2, padding=1)
        self.encoder5 = self.conv_block(512, 512, kernel_size=4, stride=2, padding=1)
        self.encoder6 = self.conv_block(512, 512, kernel_size=4, stride=2, padding=1)
        self.encoder7 = self.conv_block(512, 512, kernel_size=4, stride=2, padding=1)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.decoder1 = self.deconv_block(512, 512, kernel_size=4, stride=2, padding=1)
        self.decoder2 = self.deconv_block(1024, 512, kernel_size=4, stride=2, padding=1)
        self.decoder3 = self.deconv_block(1024, 512, kernel_size=4, stride=2, padding=1)
        self.decoder4 = self.deconv_block(1024, 512, kernel_size=4, stride=2, padding=1, dropout=False)
        self.decoder5 = self.deconv_block(1024, 256, kernel_size=4, stride=2, padding=1, dropout=False)
        self.decoder6 = self.deconv_block(512, 128, kernel_size=4, stride=2, padding=1, dropout=False)
        self.decoder7 = self.deconv_block(256, 64, kernel_size=4, stride=2, padding=1, dropout=False)
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def deconv_block(self, in_channels, out_channels, kernel_size, stride, padding, dropout=True):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        b = self.bottleneck(e7)

        d1 = torch.cat([self.decoder1(b), e7], 1)
        d2 = torch.cat([self.decoder2(d1), e6], 1)
        d3 = torch.cat([self.decoder3(d2), e5], 1)
        d4 = torch.cat([self.decoder4(d3), e4], 1)
        d5 = torch.cat([self.decoder5(d4), e3], 1)
        d6 = torch.cat([self.decoder6(d5), e2], 1)
        d7 = torch.cat([self.decoder7(d6), e1], 1)
        
        return self.final_layer(d7)

def process_frame(model, frame, device, transform):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_tensor = (output_tensor * 0.5) + 0.5
    output_tensor = output_tensor.squeeze(0).cpu()
    output_array = output_tensor.permute(1, 2, 0).numpy()
    output_frame = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
    output_frame = (output_frame * 255).astype(np.uint8)
    
    return output_frame

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = Generator().to(device)
    model_path = "generator.pth"

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        print("Please ensure the pre-trained model 'generator.pth' is in the same directory.")
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully.")

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        cap = cv2.VideoCapture(0) 
        if not cap.isOpened():
            print("Error: Could not open camera.")
            exit()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting...")
                break

            denoised_frame = process_frame(model, frame, device, transform)

            resized_original = cv2.resize(frame, (256, 256))

            combined_display = np.hstack([resized_original, denoised_frame])

            cv2.imshow('Original vs. Denoised - Press Q to Quit', combined_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()