import os
import random
from glob import glob
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets
from IPython.display import clear_output
from PIL import ImageColor, Image, ImageDraw
import torch
import torchvision
import torch.nn as nn
# from torchsummary import summary

from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, AdamW
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from transformers import CLIPTokenizer, CLIPTextModel
from transformers import CLIPProcessor, CLIPModel

import lpips
from tqdm import tqdm
from torchvision.utils import make_grid
from unet_cond import Unet
from vae import VAE
from torch.optim.lr_scheduler import MultiStepLR
from linear_noise_scheduler import LinearNoiseScheduler
from loader import load_and_transform_dataset
# import loader
from facenet_pytorch import InceptionResnetV1


class Config:
    """Training configuration"""
    Image_size = 128
    batch_size = 16
    time_steps = 500
    num_epochs = 400
    scale_factor = 1
    
    # Data_dir = "/home/mali/Mali/Dataset"
    MTF_Train_data_dir = "Data/train/image/"
    MTF_Text_data_dir = "Data/train/prompt/"
    
    dataset_name_pretrained_Diff = './models'
    dataset_name_pretrained_VAE = './models'
    dataset_name = 'MTF_training_logs' # the folder of saving the generated result
    # dataset_name = 'Celebahq_MTF_4_32_32_Batch64_Latent_ImgTextTraining_textProj_2_IdentityLoss_ImageRandomWeight'
    
    clip_model_name = "openai/clip-vit-large-patch14"
    
    lr = 1e-5
    weight_decay = 1e-5

def clean_prompt(raw_prompt, wrap=False):
    """
    Cleans a raw prompt by removing commas and underscores, 
    lowercasing, and optionally wrapping in a descriptive phrase.
    """
    raw_prompt = raw_prompt.lower()
    raw_prompt = raw_prompt.replace("male", "man").replace("female", "woman")
    cleaned = raw_prompt.replace("_", " ").lower()
    cleaned = " ".join(cleaned.split())
    return f"a portrait of a {cleaned}" if wrap else cleaned


def get_loss_weights(epoch, num_epochs):
    """Calculate dynamic loss weights based on training progress"""
    progress = epoch / num_epochs
    mse_weight = 0.7 * (1 - progress) + 0.3 * progress
    l1_weight = 0.3 * (1 - progress) + 0.2 * progress
    clip_weight = 1.0 + 1.5 * progress
    lpips_weight = 0.1 + 0.1 * progress
    identity_weight = 1.5 * progress
    return mse_weight, l1_weight, clip_weight, lpips_weight, identity_weight


def adaptive_weights(epoch, ramp_up=30, peak_img_w=0.5, sustain_until=70, text_only_prob=0.3):
    """Calculate adaptive text and image weights"""
    if epoch < ramp_up:
        progress = epoch / ramp_up
        base_img_weight = 0.05 + (peak_img_w - 0.05) * progress
    elif epoch < sustain_until:
        base_img_weight = peak_img_w
    else:
        base_img_weight = max(0.35, peak_img_w * 0.8)

    if random.random() < text_only_prob:
        image_weight = 0.0
    else:
        jitter = random.uniform(-0.05, 0.05)
        image_weight = min(max(base_img_weight + jitter, 0.0), 0.5)

    text_weight = 1.0 - image_weight
    return text_weight, image_weight


def tensor_to_image(xt: torch.Tensor) -> Image:
    """Revert the transformations on the tensor and return corresponding Image"""
    xt = xt.cpu()
    if len(xt.shape) == 4 and xt.shape[0] == 1:
        xt = xt.squeeze()
    
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])
    return reverse_transforms(xt)


def plot_random_train_images(image_path, text_path, n_images=10):
    """Show random training images with their prompts"""
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 5))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        index = random.randint(0, len(image_path) - 1)
        image = Image.open(image_path[index])
        with open(text_path[index], "r", encoding="utf-8") as f:
            text = f.read().strip()
        ax.set_title(text[:25] + ': ' + str(len(text)), fontsize=10)
        ax.imshow(image)
        plt.tight_layout()
    plt.show()


def show_processed_images(dataset, tokenizer, num_images=10):
    """Show processed images from dataset"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for i in range(num_images):
        idx = random.randint(0, len(dataset) - 1)
        img_tensor, tokenized_text, text_orig, _, _, _ = dataset[idx]
        img = tensor_to_image(img_tensor)
        input_ids = tokenized_text["input_ids"]
        text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        axes[i].set_title(text[:25] + ': ' + str(len(text)), fontsize=10)
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.show()
    plt.close()


def sample(model, scheduler, vae, tokenizer, textencoder, device, config, save_monitoring_path, epoch_idx=None):
    """Generate samples from the model"""
    model.eval()
    time_steps = scheduler.num_timesteps
    im_size = config.Image_size // 2 ** sum([True, True])
    xt = 0.5 * torch.randn((1, 4, im_size, im_size)).to(device)
    
    prompt = [
        "White, Female, Young",
        "Asian_indian, Male, Young",
        "Asian_chinese_korean, Female, Young",
        "Asian_indian, Male, Old",
        "Black, Male, Young"
    ]
    
    random_integer = random.randint(0, len(prompt)-1)
    print(f'The prompt chose is: {random_integer}')
    
    cleaned_prompt = clean_prompt(prompt[random_integer])
    tokenized_text = tokenizer(cleaned_prompt, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
    indexed_tokens = tokenized_text['input_ids']
    att_mask = tokenized_text['attention_mask']
    
    with torch.no_grad():
        text_embeddings = textencoder(indexed_tokens.to(device), attention_mask=att_mask.to(device)).last_hidden_state

    for i in tqdm(reversed(range(time_steps))):
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device), text_embeddings.to(device))
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        if i % 50 == 0 or i == 0:
            ims = vae.to(device).decode(xt.float() / config.scale_factor)
        else:
            ims = xt
        
        if not os.path.exists(os.path.join(config.dataset_name, f'samples_txt_{time_steps}')):
            if not os.path.exists(config.dataset_name):
                os.makedirs(config.dataset_name, exist_ok=True)
            os.makedirs(os.path.join(config.dataset_name, f'samples_txt_{time_steps}'), exist_ok=True)
            
        if i % 50 == 0 or i == 0:
            ims = torch.clamp(ims, -1., 1.).detach().cpu()
            ims = (ims + 1) / 2
            ims = ims * 255
            ims = ims.to(torch.uint8)
            
            grid = make_grid(ims, nrow=2)
            img = torchvision.transforms.ToPILImage()(grid)
            img.save(os.path.join(config.dataset_name, f'samples_txt_{time_steps}', 'x0_{}.png'.format(i)))
            if i == 0:
                img.save(os.path.join(save_monitoring_path, f'Txt_epoch_{epoch_idx}_{random_integer}.png'))
            img.close()


def load_data(config):
    """Load and prepare training data"""
    print('Data path:', config.MTF_Text_data_dir)

    
    
    image_path = sorted(glob(os.path.join(config.MTF_Train_data_dir, '*.*')))
    text_path = sorted(glob(os.path.join(config.MTF_Text_data_dir, '*.*')))
    
    print(f"Total samples - Images: {len(image_path)}, Text: {len(text_path)}")
    print('Image path samples:', image_path[:2])
    print('Text path samples:', text_path[:2])
    
    return image_path, text_path


def setup_models(config, device):
    """Initialize all models and components"""
    print("Setting up models...")
    
    # Load CLIP models
    tokenizer = CLIPTokenizer.from_pretrained(config.clip_model_name)
    processor = CLIPProcessor.from_pretrained(config.clip_model_name)
    textencoder = CLIPTextModel.from_pretrained(config.clip_model_name).to(device)
    clip_model = CLIPModel.from_pretrained(config.clip_model_name, output_hidden_states=True).to(device)
    
    # Freeze CLIP weights
    for param in clip_model.parameters():
        param.requires_grad = False
    for param in textencoder.parameters():
        param.requires_grad = False
    print("Loaded CLIP models successfully!")
    
    # Initialize VAE and Unet
    vae = VAE(im_channels=3).to(device)
    model = Unet(im_channels=4).to(device)
    
    # Initialize noise scheduler
    scheduler = LinearNoiseScheduler(
        num_timesteps=config.time_steps,
        beta_start=1e-4,
        beta_end=0.02,
        ldm_scheduler=True
    )
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.95))
    lr_scheduler = MultiStepLR(optimizer, milestones=[10, 25, 50, 75, 100, 125, 150, 175, 250, 300], gamma=0.5)
    
    # Initialize loss functions
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    
    # Initialize identity model
    identity_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    return {
        'tokenizer': tokenizer,
        'processor': processor,
        'textencoder': textencoder,
        'clip_model': clip_model,
        'vae': vae,
        'model': model,
        'scheduler': scheduler,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler,
        'mse_loss': mse_loss,
        'l1_loss': l1_loss,
        'lpips_model': lpips_model,
        'identity_model': identity_model
    }


def load_checkpoints(config, device, models_dict):
    """Load pre-trained checkpoints"""
    vae = models_dict['vae']
    model = models_dict['model']
    optimizer = models_dict['optimizer']
    lr_scheduler = models_dict['lr_scheduler']
    
    start_epoch = 0
    
    # Load VAE checkpoint
    vae_path = os.path.join(config.dataset_name_pretrained_VAE, 'vae_autoencoder.pth')
    if os.path.exists(vae_path):
        vae_state = torch.load(vae_path, map_location=device)
        # Remove 'module.' prefix if present (from DDP/DataParallel)
        if list(vae_state.keys())[0].startswith('module.'):
            vae_state = {k.replace('module.', ''): v for k, v in vae_state.items()}
        vae.load_state_dict(vae_state)
        vae.eval()
        vae.float()
        for param in vae.parameters():
            param.requires_grad = False
        print('--Loaded VAE checkpoint--')
    
    # Load Diffusion model checkpoint
    diff_path = os.path.join(config.dataset_name_pretrained_Diff, f'ddpm_{config.time_steps}_light.pth')
    if os.path.exists(diff_path):
        checkpoint = torch.load(diff_path, map_location=device)
        
        # Remove 'module.' prefix from model state dict if present
        model_state = checkpoint["model_state_dict"]
        if list(model_state.keys())[0].startswith('module.'):
            model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
        
        model.load_state_dict(model_state)
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # start_epoch = checkpoint["epoch"] + 1
        
        # if start_epoch > 0:
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = config.lr
        # print(f"Resuming training from epoch {start_epoch} on {device}...")
    else:
        print("No diffusion model found, starting from scratch")
    
    return start_epoch


def setup_monitoring(config, start_epoch):
    """Setup monitoring directories and load/initialize metrics"""
    save_monitoring_path = os.path.join(config.dataset_name, "monitoring")
    if not os.path.exists(save_monitoring_path):
        if not os.path.exists(config.dataset_name):
            os.makedirs(config.dataset_name, exist_ok=True)
        os.makedirs(save_monitoring_path, exist_ok=True)
    
    # Setup metrics
    metrics_path = os.path.join(save_monitoring_path, "metrics.csv")
    if os.path.exists(metrics_path) and start_epoch > 0:
        metrics = pd.read_csv(metrics_path)
        print("Loaded previous metrics!")
    else:
        metrics = pd.DataFrame(columns=['epoch', 'mse_loss', 'l1_loss', 'clip_loss', 'lpips_loss', 'identity_loss', 'total_loss'])
        print('New metrics file created!')
    
    # Setup loss history
    best_loss_path = os.path.join(save_monitoring_path, "best_loss_history.npy")
    if os.path.exists(best_loss_path) and start_epoch > 0:
        best_loss_history = np.load(best_loss_path).tolist()
        best_loss = best_loss_history[-1]
        print('Best_loss_history loaded!')
    else:
        best_loss_history = []
        best_loss = np.inf
        print('Best_loss_history created!')
    
    loss_path = os.path.join(save_monitoring_path, "loss_history.npy")
    if os.path.exists(loss_path) and start_epoch > 0:
        loss_history = np.load(loss_path).tolist()
    else:
        loss_history = []
    
    return {
        'save_monitoring_path': save_monitoring_path,
        'metrics': metrics,
        'metrics_path': metrics_path,
        'best_loss': best_loss,
        'best_loss_history': best_loss_history,
        'loss_history': loss_history
    }


def train_epoch(epoch_idx, config, device, models_dict, train_dataloader, monitoring_dict):
    """Train for one epoch"""
    model = models_dict['model']
    optimizer = models_dict['optimizer']
    scheduler = models_dict['scheduler']
    mse_loss = models_dict['mse_loss']
    l1_loss = models_dict['l1_loss']
    lpips_model = models_dict['lpips_model']
    identity_model = models_dict['identity_model']
    vae = models_dict['vae']
    textencoder = models_dict['textencoder']
    clip_model = models_dict['clip_model']
    processor = models_dict['processor']
    
    # Reinitialize scheduler for each epoch
    scheduler = LinearNoiseScheduler(
        num_timesteps=config.time_steps,
        beta_start=1e-4,
        beta_end=0.02,
        ldm_scheduler=True
    )
    
    print(f"\nEpoch {epoch_idx + 1}/{config.num_epochs} - Using {config.time_steps} timesteps")
    
    model.train()
    losses, losses_mse, losses_l1, losses_clip, losses_lpips, losses_identity = [], [], [], [], [], []
    
    # Get loss weights for this epoch
    alpha_mse, beta_l1, gamma_clip, delta_lpips, epsilon_id = get_loss_weights(epoch_idx, config.num_epochs)
    
    step_count = monitoring_dict.get('step_count', 0)
    progress_bar = tqdm(train_dataloader)
    
    # Initialize optimizer gradients
    optimizer.zero_grad()
    
    for batch_idx, (im, _, txt_original, img_name, indexed_tokens, att_mask) in enumerate(progress_bar):
        text_weight, image_weight = adaptive_weights(epoch_idx)
        
        im = im.float().to(device)
        
        with torch.no_grad():
            txt_embed = textencoder(indexed_tokens.to(device), attention_mask=att_mask.to(device)).last_hidden_state
            latents, _ = vae.encode(im)
        
        noise = torch.randn_like(latents).to(device)
        t = torch.randint(0, config.time_steps, (latents.shape[0],)).to(device)
        noisy_im = scheduler.add_noise(latents, noise, t)
        
        # Determine if using text-only mode
        use_text_only = epoch_idx < -1
        
        if use_text_only:
            noise_pred = model(noisy_im, t, txt_embed)
        else:
            im_for_clip = (im.clone().detach() + 1) / 2
            im_for_clip = im_for_clip.clamp(0, 1)
            
            inputs = processor(text=txt_original, images=im_for_clip, return_tensors="pt", truncation=True, padding=True, do_rescale=False).to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            pixel_values = F.interpolate(im_for_clip, size=(224, 224), mode='bilinear', align_corners=False)
            
            with torch.no_grad():
                outputs = clip_model(**inputs)
                image_embeds = outputs.image_embeds
                vision_outputs = clip_model.vision_model(pixel_values=pixel_values, output_hidden_states=True)
                img_hidden_states = vision_outputs.hidden_states[-1]
            
            image_emb = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            noise_pred = model(noisy_im, t, txt_embed, img_hidden_states, text_weight, image_weight)
        
        # Compute losses
        loss_mse_val = mse_loss(noise_pred, noise)
        loss_l1_val = torch.abs(noise_pred - noise).mean()
        
        with torch.no_grad():
            img_pred = vae.decode(noise_pred.float())
            noise_target = vae.decode(noise.float())
        
        img_pred = 2 * img_pred - 1
        noise_target = 2 * noise_target - 1
        loss_lpips_val = lpips_model(img_pred, noise_target).mean()
        
        compute_clip_loss = (batch_idx % 16 == 0)
        if compute_clip_loss:
            with torch.no_grad():
                idx = torch.randint(0, latents.size(0), (2,))
                single_latent = latents[idx]
                recon_im = vae.decode(single_latent.float())
                recon_im = (recon_im.clamp(-1, 1) + 1) / 2
                
                clip_inputs = processor(images=recon_im, return_tensors="pt", do_rescale=False)
                clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}
                
                gen_clip_emb = clip_model.get_image_features(pixel_values=clip_inputs["pixel_values"])
                gen_clip_emb = gen_clip_emb / gen_clip_emb.norm(dim=-1, keepdim=True)
                
                original_for_id = (im[idx].clamp(-1, 1) + 1) / 2
                id_embed_gen = identity_model(recon_im)
                id_embed_target = identity_model(original_for_id)
                identity_loss = 1 - F.cosine_similarity(id_embed_gen, id_embed_target, dim=-1).mean()
            
            if use_text_only:
                text_inputs = processor.tokenizer(txt_original, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
                with torch.no_grad():
                    target_clip_emb = clip_model.get_text_features(
                        input_ids=text_inputs.input_ids[idx],
                        attention_mask=text_inputs.attention_mask[idx]
                    )
            else:
                target_clip_emb = image_emb[idx]
            
            target_clip_emb = target_clip_emb / target_clip_emb.norm(dim=-1, keepdim=True)
            clip_loss = 1 - F.cosine_similarity(gen_clip_emb, target_clip_emb, dim=-1).mean()
        else:
            clip_loss = torch.tensor(0.0, device=device)
            identity_loss = torch.tensor(0.0, device=device)
        
        # Final loss
        loss = alpha_mse * loss_mse_val + beta_l1 * loss_l1_val + gamma_clip * clip_loss + delta_lpips * loss_lpips_val + epsilon_id * identity_loss
        
        # Track losses
        losses_mse.append(loss_mse_val.item())
        losses_l1.append(loss_l1_val.item())
        losses_clip.append(clip_loss.item())
        losses_lpips.append(loss_lpips_val.item())
        losses_identity.append(identity_loss.item())
        losses.append(loss.item())
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        step_count += 1
        
        progress_bar.set_description(f'Epoch: {epoch_idx+1} | Step: {step_count}')
    
    print('Finished epoch:{} | MSE Loss:{:.4f} | L1 Loss:{:.4f} | CLIP Loss:{:.4f} | Lpips Loss:{:.4f} | Identity loss:{:.4f} | Total Loss:{:.4f}'.format(
        epoch_idx + 1,
        np.mean(losses_mse),
        np.mean(losses_l1),
        np.mean(losses_clip),
        np.mean(losses_lpips),
        np.mean(losses_identity),
        np.mean(losses)
    ))
    
    monitoring_dict['step_count'] = step_count
    
    return {
        'losses': losses,
        'losses_mse': losses_mse,
        'losses_l1': losses_l1,
        'losses_clip': losses_clip,
        'losses_lpips': losses_lpips,
        'losses_identity': losses_identity
    }


def save_checkpoint(epoch_idx, config, models_dict, monitoring_dict, epoch_losses):
    """Save checkpoints and update metrics"""
    model = models_dict['model']
    optimizer = models_dict['optimizer']
    lr_scheduler = models_dict['lr_scheduler']
    
    save_monitoring_path = monitoring_dict['save_monitoring_path']
    loss_history = monitoring_dict['loss_history']
    best_loss = monitoring_dict['best_loss']
    best_loss_history = monitoring_dict['best_loss_history']
    metrics = monitoring_dict['metrics']
    metrics_path = monitoring_dict['metrics_path']
    
    lr_scheduler.step()
    mean_loss = np.mean(epoch_losses['losses'])
    loss_history.append(mean_loss)
    np.save(os.path.join(save_monitoring_path, "loss_history.npy"), np.array(loss_history))
    
    checkpoint = {
        # "epoch": epoch_idx,
        "model_state_dict": model.state_dict(),
        # "optimizer_state_dict": optimizer.state_dict(),
        # "scheduler_state_dict": lr_scheduler.state_dict(),
    }
    
    if epoch_idx % 5 == 0 or epoch_idx + 1 == config.num_epochs:
        torch.save(checkpoint, os.path.join(config.dataset_name, f'chk_ddpm_{config.time_steps}.pth'))
        print("Checkpoint saved successfully!")
    
    if mean_loss < best_loss:
        best_loss = mean_loss
        best_loss_history.append(best_loss)
        np.save(os.path.join(save_monitoring_path, "best_loss_history.npy"), np.array(best_loss_history))
        torch.save(checkpoint, os.path.join(config.dataset_name, f'best_ddpm_{config.time_steps}.pth'))
        print("Best model saved!")
    
    new_row = {
        'epoch': epoch_idx + 1,
        'mse_loss': round(np.mean(epoch_losses['losses_mse']), 4),
        'l1_loss': round(np.mean(epoch_losses['losses_l1']), 4),
        'clip_loss': round(np.mean(epoch_losses['losses_clip']), 4),
        'lpips_loss': round(np.mean(epoch_losses['losses_lpips']), 4),
        'identity_loss': round(np.mean(epoch_losses['losses_identity']), 4),
        'total_loss': round(mean_loss, 4),
    }
    metrics = pd.concat([metrics, pd.DataFrame([new_row])], ignore_index=True)
    metrics.to_csv(metrics_path, index=False)
    
    monitoring_dict['best_loss'] = best_loss
    monitoring_dict['best_loss_history'] = best_loss_history
    monitoring_dict['loss_history'] = loss_history
    monitoring_dict['metrics'] = metrics

def main():
    """Main training function"""
    # Initialize configuration
    config = Config()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    image_path, text_path = load_data(config)
    
    # Optionally visualize some training data
    # plot_random_train_images(image_path, text_path)
    
    # Setup all models
    models_dict = setup_models(config, device)
    
    # Load checkpoints
    start_epoch = load_checkpoints(config, device, models_dict)
    
    # Create dataset and dataloader
    train_dataset, tokenizer, textencoder = load_and_transform_dataset(
        config.Image_size, image_path, text_path, 
        models_dict['tokenizer'], models_dict['textencoder'], 
        use_augmentation=True
    )
    print('train_dataset:', len(train_dataset))
    
    # Set num_workers=0 for Windows to avoid pickle errors with lambda transforms
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True
    )
    
    # Optionally show processed images
    # show_processed_images(train_dataset, tokenizer)
    
    # Setup monitoring
    monitoring_dict = setup_monitoring(config, start_epoch)
    monitoring_dict['step_count'] = 0
    
    print("--DDPM start training--")
    
    # Training loop
    for epoch_idx in range(start_epoch, config.num_epochs):
        epoch_losses = train_epoch(
            epoch_idx, config, device, models_dict, 
            train_dataloader, monitoring_dict
        )
        
        save_checkpoint(epoch_idx, config, models_dict, monitoring_dict, epoch_losses)
    
    print('Done Training!')


if __name__ == "__main__":
    main()
