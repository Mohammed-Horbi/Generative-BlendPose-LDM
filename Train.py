import os
import time
import torch
import random
import lpips
import argparse

import numpy as np
import torch.nn as nn
import torchvision
from tqdm import tqdm
from glob import glob

import torch.nn.functional as F
from collections import OrderedDict

from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel 
from transformers import CLIPTokenizer, CLIPTextModel

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim import Adam, AdamW
import torch.nn.functional as F
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
from torch.optim.lr_scheduler import MultiStepLR

from vae import VAE
from unet_cond import Unet
from linear_noise_scheduler import LinearNoiseScheduler

import loader

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

def get_loss_weights(epoch, num_epochs):
    progress = epoch / num_epochs
    mse_weight = 0.7 * (1 - progress) + 0.3 * progress
    l1_weight = 0.3 * (1 - progress) + 0.2 * progress
    clip_weight = 1.0 + 1.5 * progress
    lpips_weight = 0.1 + 0.1 * progress
    identity_weight = 1.5 * progress  # Ramp from 0.0 to 1.0
    return mse_weight, l1_weight, clip_weight, lpips_weight, identity_weight

def adaptive_weights(epoch,
                     ramp_up=30,
                     peak_img_w=0.5,
                     sustain_until=70,
                     text_only_prob=0.3):  # 30% of batches are text-only
    # Epoch-based base image weight schedule
    if epoch < ramp_up:
        progress = epoch / ramp_up
        base_img_weight = 0.05 + (peak_img_w - 0.05) * progress
    elif epoch < sustain_until:
        base_img_weight = peak_img_w
    else:
        base_img_weight = max(0.35, peak_img_w * 0.8)

    # Occasionally switch to text-only
    if random.random() < text_only_prob:
        image_weight = 0.0
    else:
        # Add jitter to prevent overfitting to fixed ratio
        jitter = random.uniform(-0.05, 0.05)
        image_weight = min(max(base_img_weight + jitter, 0.0), 0.5)

    text_weight = 1.0 - image_weight
    return text_weight, image_weight

def main():
    Diff_model = './models' 
    VAE_model = './models' 
    dataset_name = 'MTF_training_logs' # the folder of saving the generated result
    Image_size = 128
    batch_size = 32
    time_steps = 500 
    num_epochs = 350
    # scale_factor = 1
    num_steps = time_steps  # put the same value for the time_steps and num_steps

    MTF_Image_data_dir = "Data/train/image/"
    MTF_Text_data_dir = "Data/train/prompt/"
    # dir_path = Data_dir
    print('Data path: ', MTF_Text_data_dir)

    clip_model_name = "openai/clip-vit-large-patch14"
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    textencoder = CLIPTextModel.from_pretrained(clip_model_name).to(device) 
    clip_model = CLIPModel.from_pretrained(clip_model_name, output_hidden_states=True).to(device) 

    for param in clip_model.parameters():
        param.requires_grad = False
    for param in textencoder.parameters():
        param.requires_grad = False
    print("--CLIP models loaded successfully!--")

    vae = VAE(im_channels=3).to(device)  
    # vae = vae.float()
    model = Unet(im_channels=4).to(device)
    # model = DDP(model, device_ids=[local_rank])
    # model = DDP(
    #     model,
    #     device_ids=[local_rank],
    #     output_device=local_rank,
    #     find_unused_parameters=True
    # )
    scheduler = LinearNoiseScheduler(num_timesteps=time_steps, #diffusion_config['num_timesteps'],
                                beta_start=1e-4, #diffusion_config['beta_start'],
                                beta_end=0.02, #diffusion_config['beta_end'],
                                ldm_scheduler=True)
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5, betas=(0.9, 0.95))
    lr_scheduler = MultiStepLR(optimizer, milestones=[10, 25, 50, 75, 100, 125, 150, 175, 250, 300], gamma=0.5)
    mse_loss = nn.MSELoss()
    # l1_loss = nn.L1Loss()
    # if local_rank == 0: it is not needed here
    lpips_model = lpips.LPIPS(net='vgg').to(device)

    # Load the pre-traiened the VAE model
    if os.path.exists(os.path.join(VAE_model,'vae_autoencoder.pth')):        
        vae.load_state_dict(torch.load(os.path.join(VAE_model, 'vae_autoencoder.pth'), map_location=device))
        vae.eval()                   
        vae.float()                  
        for param in vae.parameters():
            param.requires_grad = False                
        print('--VAE loaded successfully--')
    else:
        print("There is no checkpoint of vae loaded")

    if os.path.exists(os.path.join(Diff_model, 'ddpm_500.pth')):
        checkpoint = torch.load(os.path.join(Diff_model, 'ddpm_500.pth'), map_location='cpu')
        # Restore the model, optimizer, and scheduler states
        # if local_rank == 0:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # Resume training from the saved epoch
        start_epoch = checkpoint["epoch"] + 1  # Continue from the next epoch
        
        # start_epoch = 0
        if start_epoch > 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 1e-5  # Your new learning rate
        print(f"Resuming training from epoch {start_epoch} but for transfering it will start from {start_epoch} on {device}...")
    else:
        # optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
        # if local_rank == 0:
        print("There is no checkpoint of diffusion model loaded")
        start_epoch = 0
    start_epoch

    image_path = sorted(glob(os.path.join(MTF_Image_data_dir, '*.*')))
    text_path = sorted(glob(os.path.join(MTF_Text_data_dir, '*.*')))

    image_path = sorted(list(image_path))
    text_path = sorted(list(text_path))

    print(f"Total samples of images: {len(image_path)}, Text samples: {len(text_path)}")
    # N_train = len(image_path)
    print('image_path: ', image_path[:1])
    print('text_path: ',text_path[:1])

    # Load dataset
    train_dataset, tokenizer, textencoder = loader.load_and_transform_dataset(Image_size, image_path, text_path, tokenizer, textencoder)
    print('train_dataset: ', len(train_dataset))  # Check the number of images
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    identity_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    # if local_rank == 0:
    print("--DDPM start training--")    
    best_loss_path = os.path.join(Save_monitoring_path, "best_loss_history.npy")
    if os.path.exists(best_loss_path) and start_epoch > 0:
        best_loss_history = np.load(best_loss_path).tolist()
        best_loss = best_loss_history[-1]
        print('Best_loss_history loaded!')
    else:
        best_loss_history = []
        best_loss = np.inf
        print('Best_loss_history created!')

    loss_path = os.path.join(Save_monitoring_path, "loss_history.npy")

    if os.path.exists(loss_path) and start_epoch > 0:
        loss_history = np.load(loss_path).tolist()
    else:
        loss_history = []

    # gradient_accumulation_steps = 2  # Adjust this value as needed
    # effective_batch_size = batch_size * gradient_accumulation_steps
    step_count = 0

    Save_monitoring_path = os.path.join(dataset_name, "monitoring")
    if not os.path.exists(Save_monitoring_path):  
        if not os.path.exists(dataset_name):  
            os.makedirs(dataset_name, exist_ok=True)
        os.makedirs(Save_monitoring_path, exist_ok=True)

    for epoch_idx in range(start_epoch, num_epochs):
        # Update timesteps for the current epoch
        # time_steps = get_dynamic_timesteps(epoch_idx)
        scheduler = LinearNoiseScheduler(num_timesteps=time_steps,
                                    beta_start=1e-4,
                                    beta_end=0.02,
                                    ldm_scheduler=True)
        
        model.train()
        losses, losses_mse, losses_l1, losses_clip, losses_lpips, losses_identity = [], [], [], [], [], []
        
        # Get loss weights once per epoch as these don't need to change per batch
        alpha_mse, beta_l1, gamma_clip, delta_lpips, epsilon_id = get_loss_weights(epoch_idx, num_epochs)
        
        # Synchronize loss weights across GPUs
        # if dist.is_initialized():
        #     loss_weights = torch.tensor([alpha_mse, beta_l1, gamma_clip, delta_lpips, epsilon_id]).to(device)
        #     dist.broadcast(loss_weights, src=0)
        #     alpha_mse, beta_l1, gamma_clip, delta_lpips, epsilon_id = loss_weights.tolist()    
        
        progress_bar = tqdm(train_dataloader)
        for batch_idx, (im, _, txt_original, img_name, indexed_tokens, att_mask) in enumerate(progress_bar):                                  
            text_weight, image_weight = adaptive_weights(epoch_idx)                        
            im = im.float().to(device)
            # txt_embed = txt_embed.to(device)      
            with torch.no_grad():       
                txt_embed = textencoder(indexed_tokens.to(device), attention_mask=att_mask.to(device)).last_hidden_state
            # input_image = im #Image.open(img_name).convert("RGB")                
            with torch.no_grad():
                latents, _ = vae.encode(im)
            
            # Sample random noise
            noise = torch.randn_like(latents).to(device)

            # Sample timestep
            t = torch.randint(0, time_steps, (latents.shape[0],)).to(device)
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(latents, noise, t)
            
            # if epoch_idx < ramp_st or epoch_idx % 10 == 0:
            if epoch_idx < -1: #>=250 and epoch_idx <= 300:
                use_text_only = True
            else:
                use_text_only = False
                # conditioning = txt_embed
            if use_text_only:            
                # conditioning = txt_embed
                noise_pred = model(noisy_im, t, txt_embed)
            else:                  
                im_for_clip = (im.clone().detach() + 1) / 2
                im_for_clip = im_for_clip.clamp(0, 1)
                
                # input_image = [to_pil_image(img.cpu()) for img in im_for_clip]
                inputs = processor(text=txt_original, images=im_for_clip, return_tensors="pt", truncation=True, padding=True, do_rescale=False).to(device)           
                inputs = {k: v.to(device) for k, v in inputs.items()}
                pixel_values = F.interpolate(im_for_clip, size=(224, 224), mode='bilinear', align_corners=False)
                # pixel_values = pixel_values.to(device)
                with torch.no_grad():
                    outputs = clip_model(**inputs)
                    image_embeds = outputs.image_embeds   # Shape: (batch_size, hidden_dim)
                    vision_outputs = clip_model.vision_model(pixel_values=pixel_values, output_hidden_states=True)
                    img_hidden_states = vision_outputs.hidden_states[-1] # Shape: [1, 50, 1024]              
                image_emb = image_embeds / image_embeds.norm(dim=-1, keepdim=True)                
                noise_pred = model(noisy_im, t, txt_embed, img_hidden_states, text_weight, image_weight)            

            # Compute combined loss
            loss_mse = mse_loss(noise_pred, noise)
            loss_l1 = torch.abs(noise_pred - noise).mean()
            # print(noise_pred.shape, noise.shape)
            # Because the lpips does not work with shape like (B, 4, 32, 32) we have to convert the t=results to RGB as follows.
            with torch.no_grad():
                img_pred = vae.decode(noise_pred.float())    # → should output (B, 3, 32, 32)
                noise_target = vae.decode(noise.float())       # → same shape        
            # Normalize to [-1, 1] as LPIPS expects
            img_pred = 2 * img_pred - 1
            noise_target = 2 * noise_target - 1
            loss_lpips = lpips_model(img_pred, noise_target).mean()
            compute_clip_loss = (batch_idx % 16 == 0)
            if compute_clip_loss:
                # Decode image from latent to compute CLIP loss
                with torch.no_grad():
                    idx = torch.randint(0, latents.size(0), (2,))
                    single_latent = latents[idx]
                    recon_im = vae.decode(single_latent.float())#.unsqueeze(0))  # Only 2 images
                    # recon_im = vae.decode(latents)  # [B, 3, H, W]
                    recon_im = (recon_im.clamp(-1, 1) + 1) / 2  # [0, 1] for CLIP
                
                    recon_im = recon_im.to(device)  # Already in [0, 1] range after your clamp
                    clip_inputs = processor(
                        images=recon_im,            # Directly pass tensor batch
                        return_tensors="pt",
                        do_rescale=False            # Prevent re-normalizing [0, 1] to [0, 255]
                    )
                    clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}
                
                    # Use get_image_features instead of full forward
                    gen_clip_emb = clip_model.get_image_features(pixel_values=clip_inputs["pixel_values"])
                    gen_clip_emb = gen_clip_emb / gen_clip_emb.norm(dim=-1, keepdim=True)

                    # Identity loss: get target image for identity comparison
                    original_for_id = (im[idx].clamp(-1, 1) + 1) / 2  # Normalize original images to [0, 1]
                    id_embed_gen = identity_model(recon_im)
                    id_embed_target = identity_model(original_for_id)
                    identity_loss = 1 - F.cosine_similarity(id_embed_gen, id_embed_target, dim=-1).mean()
            
                # Target for CLIP loss
                if use_text_only:
                    # Already sanitized
                    text_inputs = processor.tokenizer(
                        txt_original,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=77
                    ).to(device)
                
                    with torch.no_grad():
                        target_clip_emb = clip_model.get_text_features(
                            input_ids=text_inputs.input_ids[idx],
                            attention_mask=text_inputs.attention_mask[idx]
                        )    
                else:
                    target_clip_emb = image_emb[idx]
                
                target_clip_emb = target_clip_emb / target_clip_emb.norm(dim=-1, keepdim=True)
                # Cosine similarity loss
                clip_loss = 1 - F.cosine_similarity(gen_clip_emb, target_clip_emb, dim=-1).mean()
            else:
                clip_loss = torch.tensor(0.0, device=device)
                identity_loss = torch.tensor(0.0, device=device)

            # Final composite loss function       
            loss = (alpha_mse * loss_mse + beta_l1 * loss_l1 + gamma_clip * clip_loss + delta_lpips * loss_lpips + epsilon_id * identity_loss) / gradient_accumulation_steps
            full_loss = alpha_mse * loss_mse + beta_l1 * loss_l1 + gamma_clip * clip_loss + delta_lpips * loss_lpips + epsilon_id * identity_loss

            # Track losses
            losses_mse.append(loss_mse.item())
            losses_l1.append(loss_l1.item())
            losses_clip.append(clip_loss.item())
            losses_lpips.append(loss_lpips.item())
            losses_identity.append(identity_loss.item())
            losses.append(full_loss.item())
            
            loss.backward()
            # Apply gradient clipping **here**
            # Step optimization every gradient_accumulation_steps or at the end of epoch
            # if ((batch_idx + 1) % gradient_accumulation_steps == 0) or (batch_idx + 1 == len(train_dataloader)):
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
                # optimizer.zero_grad()
            step_count += 1

            # Update progress bar description
            # if local_rank == 0:
            progress_bar.set_description(f'Epoch: {epoch_idx+1} | Step: {step_count}')
        # if local_rank == 0:        
        print('Finished epoch:{} | MSE Loss:{:.4f} | L1 Loss:{:.4f} | CLIP Loss:{:.4f} | Lpips Loss:{:.4f} | Identity loss:{:.4f} | Total Loss:{:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses_mse),
            np.mean(losses_l1),
            np.mean(losses_clip),
            np.mean(losses_lpips),
            np.mean(losses_identity),
            np.mean(losses)
        ))
        
        lr_scheduler.step()
        loss_history.append(np.mean(losses))
        mean_loss = np.mean(losses)
        np.save(os.path.join(Save_monitoring_path, "loss_history.npy"), np.array(loss_history))
        
        # Prepare the checkpoint
        checkpoint = {
                "epoch": epoch_idx,  # Last completed epoch
                "model_state_dict": model.state_dict(),            
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
            }
        # Save the checkpoint
        if epoch_idx % 5 == 0 or epoch_idx+1== num_epochs:            
            torch.save(checkpoint, os.path.join(dataset_name,
                                                            f'ddpm_ckpt_New_LessSteps{time_steps}.pth'))
            print("Checkpoint saved successfully!")
        
        # Save best model
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_loss_history.append(best_loss)
            np.save(os.path.join(Save_monitoring_path, "best_loss_history.npy"), np.array(best_loss_history))
            torch.save(checkpoint, os.path.join(dataset_name, f'best_ddpm_model_Finetuned_LessSteps.pth'))
            print("Best model saved!")
if __name__ == "__main__":
    main()