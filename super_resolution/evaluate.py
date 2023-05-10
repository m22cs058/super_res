import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize

def calculate_psnr(original_image, generated_image):
    mse = F.mse_loss(original_image, generated_image)
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()

def calculate_ssim(original_image, generated_image):
    original_image = original_image.squeeze().cpu().numpy()
    generated_image = generated_image.detach().squeeze().cpu().numpy()

    # Resize images to ensure they are at least 7x7
    min_size = min(original_image.shape[-2:])
    target_size = max(7, min_size)
    original_image = resize(original_image, (target_size, target_size), preserve_range=True, anti_aliasing=True)
    generated_image = resize(generated_image, (target_size, target_size), preserve_range=True, anti_aliasing=True)

    ssim = structural_similarity(original_image, generated_image, data_range=generated_image.max() - generated_image.min(), multichannel=True)
    return ssim


# Assuming you have a trainloader that provides batches of data
# trainloader: DataLoader object that loads your dataset

def evaluate(netG, trainset, device):
    psnr_scores = []
    ssim_scores = []
    for low_resolution_images, high_resolution_images in trainset:
        low_resolution_images = low_resolution_images.to(device)
        high_resolution_images = high_resolution_images.to(device)

        # Pass low-resolution images through the generator model
        generated_images = netG(low_resolution_images.unsqueeze(0))
        #generated_images = transforms.ToPILImage()(generated_images[0].cpu())
        #high_resolution_images = transforms.ToPILImage()(high_resolution_images[0].cpu())
        psnr = calculate_psnr(high_resolution_images, generated_images)
        ssim = calculate_ssim(high_resolution_images, generated_images)

        psnr_scores.append(psnr)
        ssim_scores.append(ssim)

    average_psnr = sum(psnr_scores) / len(psnr_scores)
    average_ssim = sum(ssim_scores) / len(ssim_scores)
    print("Maximum PSNR:", max(psnr_scores))
    print("Average PSNR:", average_psnr)
    print("Minimum PSNR:", min(psnr_scores))
    print("Average SSIM:", average_ssim)
