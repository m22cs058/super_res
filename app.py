import streamlit as st
from torchvision import transforms
import torch
from PIL import Image
import torch.nn as nn
import math
import io

class ResidualBlock(nn.Module):
  def __init__(self, channels):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(channels)
    self.prelu = nn.PReLU()
    self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(channels)
  def forward(self, x):
    residual = self.conv1(x)
    residual = self.bn1(residual)
    residual = self.prelu(residual)
    residual = self.conv2(residual)
    residual = self.bn2(residual)
    return x + residual

class UpsampleBlock(nn.Module):
  def __init__(self, in_channels, up_scale):
    super(UpsampleBlock, self).__init__()
    self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, 
                          kernel_size=3, padding=1)
    self.pixel_shuffle = nn.PixelShuffle(up_scale)
    self.prelu = nn.PReLU()
  def forward(self, x):
    x = self.conv(x)
    x = self.pixel_shuffle(x)
    x = self.prelu(x)
    return x

class Generator(nn.Module):
  def __init__(self, scale_factor):
    super(Generator, self).__init__()
    upsample_block_num = int(math.log(scale_factor, 2))

    self.block1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=9, padding=4),
        nn.PReLU()
    )

    self.block2 = ResidualBlock(64)
    self.block3 = ResidualBlock(64)
    self.block4 = ResidualBlock(64)
    self.block5 = ResidualBlock(64)
    self.block6 = ResidualBlock(64)
    self.block7 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64)
    )
    block8 = [UpsampleBlock(64, 2) for _ in range(upsample_block_num)]
    block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
    self.block8 = nn.Sequential(*block8)
  def forward(self, x):
    block1 = self.block1(x)
    block2 = self.block2(block1)
    block3 = self.block3(block2)
    block4 = self.block4(block3)
    block5 = self.block5(block4)
    block6 = self.block6(block5)
    block7 = self.block7(block6)
    block8 = self.block8(block1 + block7)
    return (torch.tanh(block8) + 1) / 2

# Load the pre-trained model
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator(4)
netG.load_state_dict(torch.load('netG2_epoch1000.pt', map_location=device))
netG.to(device)
netG.eval()

# Define the image transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# Define the Streamlit app
def app():
    st.title("Super Resolution App")
    st.write("Upload an image and generate a super resolution image.")
    
    # Upload an image
    file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    # When an image is uploaded
    if file is not None:
        # Read the uploaded image
        img = Image.open(file).convert('RGB')
        #img = img.resize(img.size[0]//4, img.size[1]//4)
        
        # Display the original image
        st.image(img, caption="Original Image", use_column_width=True)
        
        # Preprocess the image
        img = transform(img).unsqueeze(0)
        img = img.to(device)
        
        # Generate the super resolution image
        with torch.no_grad():
            sr_img = netG(img)
            
        # Convert the output tensor to a PIL Image and save the result
        sr_img = transforms.ToPILImage()(sr_img[0].cpu())
        sr_img_bytes = io.BytesIO()
        sr_img.save(sr_img_bytes, format='JPEG')
        
        # Display the super resolution image
        st.image(sr_img, caption="Super Resolution Image", use_column_width=True)
        
        # Add a download button to download the super resolution image
        st.download_button(
            label="Download Super Resolution Image",
            data=sr_img_bytes.getvalue(),
            file_name="super_resolution_image.jpg",
            mime="image/jpeg"
        )

if __name__ == '__main__':
    app()
