import streamlit as st
from torchvision import transforms
import torch
from PIL import Image
import torch.nn as nn
import math
import io
from models import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

netG = Generator(4)
netG.to(device)

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
        img = img.resize((img.size[0]//2, img.size[1]//2))
        
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
