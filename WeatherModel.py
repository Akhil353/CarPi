import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from PIL import Image
import os

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(3, 64, kernel_size=4, stride=2, padding=1, batch_norm=False)
        self.encoder2 = self.conv_block(64, 128, kernel_size=4, stride=2, padding=1)
        self.encoder3 = self.conv_block(128, 256, kernel_size=4, stride=2, padding=1)
        self.encoder4 = self.conv_block(256, 512, kernel_size=4, stride=2, padding=1)
        self.encoder5 = self.conv_block(512, 512, kernel_size=4, stride=2, padding=1)
        self.encoder6 = self.conv_block(512, 512, kernel_size=4, stride=2, padding=1)
        self.encoder7 = self.conv_block(512, 512, kernel_size=4, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.decoder1 = self.deconv_block(512, 512, kernel_size=4, stride=2, padding=1)
        self.decoder2 = self.deconv_block(1024, 512, kernel_size=4, stride=2, padding=1)
        self.decoder3 = self.deconv_block(1024, 512, kernel_size=4, stride=2, padding=1)
        self.decoder4 = self.deconv_block(1024, 512, kernel_size=4, stride=2, padding=1, dropout=False)
        self.decoder5 = self.deconv_block(1024, 256, kernel_size=4, stride=2, padding=1, dropout=False)
        self.decoder6 = self.deconv_block(512, 128, kernel_size=4, stride=2, padding=1, dropout=False)
        self.decoder7 = self.deconv_block(256, 64, kernel_size=4, stride=2, padding=1, dropout=False)

        # Final Layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        ]
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
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)

        # Bottleneck
        b = self.bottleneck(e7)

        # Decoder
        d1 = self.decoder1(b)
        d1 = torch.cat([d1, e7], 1)
        d2 = self.decoder2(d1)
        d2 = torch.cat([d2, e6], 1)
        d3 = self.decoder3(d2)
        d3 = torch.cat([d3, e5], 1)
        d4 = self.decoder4(d3)
        #d4 = torch.cat([d4, e4], 1)
        d5 = self.decoder5(d4)
        #d5 = torch.cat([d5, e3], 1)
        d6 = self.decoder6(d5)
        #d6 = torch.cat([d6, e2], 1)
        d7 = self.decoder7(d6)
        #d7 = torch.cat([d7, e1], 1)


        # Final Layer
        return self.final_layer(d7)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, target):
        x = torch.cat([input, target], 1)
        return self.main(x)
    
# --- Dataset and Dataloader ---
class ImageDataset(Dataset):
    def __init__(self, foggy_dir, ground_truth_dir, transform=None):
        self.foggy_dir = foggy_dir
        self.ground_truth_dir = ground_truth_dir
        self.transform = transform
        self.foggy_images = os.listdir(foggy_dir)

    def __len__(self):
        return len(self.foggy_images)

    def __getitem__(self, idx):
        img_name = self.foggy_images[idx]
        foggy_img_path = os.path.join(self.foggy_dir, img_name)
        ground_truth_img_path = os.path.join(self.ground_truth_dir, img_name)

        foggy_image = Image.open(foggy_img_path).convert("RGB")
        ground_truth_image = Image.open(ground_truth_img_path).convert("RGB")

        if self.transform:
            foggy_image = self.transform(foggy_image)
            ground_truth_image = self.transform(ground_truth_image)

        return foggy_image, ground_truth_image

# --- Training Script ---
def train():
    # --- Hyperparameters ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.001
    batch_size = 1
    num_epochs = 200
    
    # --- Data ---
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # NOTE: Replace with your actual data paths
    train_dataset = ImageDataset(
        foggy_dir="path/to/your/training/foggy_images",
        ground_truth_dir="path/to/your/training/ground_truth_images",
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # --- Models ---
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # --- Loss and Optimizers ---
    adversarial_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()
    
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # --- Training Loop ---
    for epoch in range(num_epochs):
        for i, (foggy_imgs, real_imgs) in enumerate(train_loader):
            foggy_imgs, real_imgs = foggy_imgs.to(device), real_imgs.to(device)

            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            
            # Real Images
            real_output = discriminator(real_imgs, foggy_imgs).view(-1)
            loss_D_real = adversarial_loss(real_output, torch.ones_like(real_output))
            
            # Fake Images
            fake_imgs = generator(foggy_imgs)
            fake_output = discriminator(fake_imgs.detach(), foggy_imgs).view(-1)
            loss_D_fake = adversarial_loss(fake_output, torch.zeros_like(fake_output))

            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()
            
            fake_output = discriminator(fake_imgs, foggy_imgs).view(-1)
            loss_G_gan = adversarial_loss(fake_output, torch.ones_like(fake_output))
            loss_G_l1 = l1_loss(fake_imgs, real_imgs) * 100 # Lambda = 100
            
            loss_G = loss_G_gan + loss_G_l1
            loss_G.backward()
            optimizer_G.step()
            
            if (i+1) % 200 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
                      f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

    # --- Save the trained model ---
    torch.save(generator.state_dict(), "generator.pth")

if __name__ == '__main__':
    train()