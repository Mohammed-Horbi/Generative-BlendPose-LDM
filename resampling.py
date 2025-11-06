import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from torchvision.utils import make_grid
from transformers import CLIPProcessor, CLIPModel 
from transformers import CLIPTokenizer, CLIPTextModel
from class_consistent import organize_images_by_class, clean_prompt
from loader import load_and_transform_dataset, ConditionalImageDataset
from functools import partialmethod


tqdm.__init__ = partialmethod(tqdm.__init__, leave=False, position=0, dynamic_ncols=True)

# Distance function
def pose_distance(p1, p2):
    return sum(abs(a - b) for a, b in zip(p1, p2))

def img_generation(model, scheduler, vae, train_dataloader, Image_size, k_blending, device, tokenizer, textencoder, processor, clip_model, time_steps, image_weight, dataset_name, Save_monitoring_path, text_weight=None):
    
    num_steps = time_steps  # set this to your desired number of steps for inferencing
    scale_factor = 1 
    
    model.eval()
    all_latents = []
    all_prompts = []
    all_image_names = []
    # Organize images by class first
    class_images = organize_images_by_class(train_dataloader)
    im_size = Image_size // 2 ** sum([True, True])
    start_batch = 0  # set your desired starting batch index
    if start_batch > 0:
        print('Save_monitoring_path::', Save_monitoring_path)
        latents_save_path = os.path.join(Save_monitoring_path, 'all_latents_with_meta.pt')
        if os.path.exists(latents_save_path):            
            data = torch.load(latents_save_path)
            all_latents.append(data['latents'])
            all_prompts.extend(data['prompts'])
            all_image_names.extend(data['image_names'])
            print(data['latents'].shape)
            print(data['prompts'][0])
            print(data['image_names'][0])
            print("Loaded existing data.")

    progress_bar = tqdm(train_dataloader)
    
    for batch_idx, (_, _, txt_original, batch_img_names, indexed_tokens, att_mask) in enumerate(progress_bar):
        # For each prompt, find a random image from the same demographic class
        batch_images = []
        if batch_idx < start_batch:
            continue
        valid_txt = []
        valid_names = []
        batch_images = []
        original_batch_tensors = []   

        for prompt, img_name in zip(txt_original, batch_img_names):
            class_key = clean_prompt(prompt, wrap=False)
            current_person_id = os.path.splitext(os.path.basename(img_name))[0].split('_')[0]

            if class_key not in class_images or not class_images[class_key]:
                print(f"[Skip] No images found for class: {class_key}")
                continue

            other_people_images = [
                (img, p, n, pid, pose) for img, p, n, pid, pose in class_images[class_key]
                if pid != current_person_id
            ]

            current_img_data = [
                (img, p, n, pid, pose) for img, p, n, pid, pose in class_images[class_key]
                if n == img_name
            ]

            if not current_img_data:
                print(f"[Skip] Pose not found for image: {img_name}")
                continue

            _, _, _, _, current_pose = current_img_data[0]
            
            if k_blending == 0:
                original_img_tensor = current_img_data[0][0].float()
                merged_img = original_img_tensor.clone().clamp(-1, 1)
            else: # Here if k_blending>=1 to merge the most similar pose/poses with the original image
                if len(other_people_images) >= k_blending:
                    # Sort by pose distance → take top k
                    similar_images = sorted(other_people_images, key=lambda x: pose_distance(x[-1], current_pose))[:k_blending]
                    # === Uniform averaging with original ===
                    imgs_stack = torch.stack([img_data[0].float() for img_data in similar_images])
                    original_img_tensor = current_img_data[0][0].float()
                    imgs_stack_with_original = torch.cat([imgs_stack, original_img_tensor.unsqueeze(0)], dim=0)
                    merged_img = imgs_stack_with_original.mean(dim=0).clamp(-1, 1)

                    # print(f"[✓] Pose-average blend of {k_blending} images + original for class {class_key}")
                elif 1 <= len(other_people_images) < k_blending:
                    similar_images = sorted(other_people_images, key=lambda x: pose_distance(x[-1], current_pose))

                    # === Uniform averaging with original ===
                    imgs_stack = torch.stack([img_data[0].float() for img_data in similar_images])
                    original_img_tensor = current_img_data[0][0].float()
                    imgs_stack_with_original = torch.cat([imgs_stack, original_img_tensor.unsqueeze(0)], dim=0)
                    merged_img = imgs_stack_with_original.mean(dim=0).clamp(-1, 1)

                    print(f"[✓] Using less images {len(similar_images)} than k_{k_blending} from another person for class {class_key}")
                else:            
                    same_person_images = [
                        (img, p, n, pid, pose) for img, p, n, pid, pose in class_images[class_key]
                        if pid == current_person_id
                    ]
                    if not same_person_images:
                        print(f"[Skip] No images found for fallback (same person) in class {class_key}")
                        continue

                    selected_images = random.sample(same_person_images, min(k_blending, len(same_person_images)))

                    # === Uniform averaging with original ===
                    imgs_stack = torch.stack([img_data[0].float() for img_data in selected_images])
                    original_img_tensor = current_img_data[0][0].float()
                    imgs_stack_with_original = torch.cat([imgs_stack, original_img_tensor.unsqueeze(0)], dim=0)
                    merged_img = imgs_stack_with_original.mean(dim=0).clamp(-1, 1)
                    print(f"[✓] Fallback pose-average blend with same person {current_person_id}")

            batch_images.append(merged_img)
            original_batch_tensors.append(original_img_tensor)  # <<< NEW: store original for this sample
            valid_txt.append(prompt)
            valid_names.append(img_name)

        if not batch_images:  # Skip this batch if we couldn't find any valid images
            continue   
        # Convert list of images to tensor
        im = torch.stack(batch_images).to(device)
        inputs = tokenizer(
            valid_txt,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            text_embeddings = textencoder(**inputs).last_hidden_state
        batch_img_names = valid_names
        txt_original = valid_txt

        # Preprocess
        image_latents = (im.clone().detach() + 1) / 2
        image_latents = image_latents.clamp(0, 1)

        xt = torch.randn(image_latents.shape[0], 4, im_size, im_size).to(device)

        inputs = processor(text=txt_original, images=image_latents, return_tensors="pt", truncation=True, padding=True, do_rescale=False).to(device)           
        inputs = {k: v.to(device) for k, v in inputs.items()}
        pixel_values = F.interpolate(image_latents, size=(224, 224), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            outputs = clip_model(**inputs)
            vision_outputs = clip_model.vision_model(pixel_values=pixel_values, output_hidden_states=True)
            img_hidden_states = vision_outputs.hidden_states[-1]
        
        # Denoising process     
        start_step = time_steps - 1  # typically 499
        step_indices = torch.linspace(start_step, 0, num_steps, dtype=torch.long)
        # _sync()
        # t0 = time.perf_counter()  
        for i in tqdm(step_indices, desc="Denoising steps:"):
            i = i.item()

            if text_embeddings.shape[0] != img_hidden_states.shape[0]:
                min_batch = min(text_embeddings.shape[0], img_hidden_states.shape[0])
                text_embeddings = text_embeddings[:min_batch]
                img_hidden_states = img_hidden_states[:min_batch]
                xt = xt[:min_batch]
                batch_img_names = batch_img_names[:min_batch]
                original_batch_tensors = original_batch_tensors[:min_batch] 
                txt_original = txt_original[:min_batch]
                print(f"⚠️ Batch size mismatch: text={text_embeddings.shape[0]}, image={img_hidden_states.shape[0]}. Truncating to {min_batch}.")                                                        
            noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device), text_embeddings.to(device), img_hidden_states, text_weight, image_weight)
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

            if i == 0:
                all_latents.append(xt.cpu())
                all_prompts.extend(txt_original)
                all_image_names.extend(batch_img_names)                            
                ims = vae.decode(xt.float() / scale_factor)
            else:
                ims = xt

            if not os.path.exists(os.path.join(dataset_name, f'samples_img_txt_{time_steps}')):
                if not os.path.exists(dataset_name): 
                    os.makedirs(dataset_name, exist_ok=True)
                os.makedirs(os.path.join(dataset_name, f'samples_img_txt_{time_steps}'), exist_ok=True)
            
            if i % 50 == 0:
                
                ims = torch.clamp(ims, -1., 1.).detach().cpu()
                ims = (ims + 1) / 2
                ims = ims * 255
                ims = ims.to(torch.uint8)            
                grid = make_grid(ims, nrow=8)
                if batch_idx==1:
                    img = torchvision.transforms.ToPILImage()(ims[0])
                    img.save(os.path.join(dataset_name, f'samples_img_txt_{time_steps}', 'x0_{}.png'.format(i)))

                if i == 0:                                                     
                    for b, img_name in enumerate(batch_img_names):
                        generated_img = torchvision.transforms.ToPILImage()(ims[b])
                        original_name = img_name
                        new_name = os.path.splitext(original_name)[0] + "_gen"
                        person_id = os.path.splitext(os.path.basename(original_name))[0].split('_')[0]
                        person_folder = os.path.join(Save_monitoring_path, 'Generated_images', person_id)
                        os.makedirs(person_folder, exist_ok=True)
                        generated_img.save(os.path.join(person_folder, f'{new_name}_{i}.png'))
                        generated_img.save(os.path.join(Save_monitoring_path, f'{new_name}_{i}.png'))
                        fig, axes = plt.subplots(1, 3, figsize=(8, 4))

                        blended_img = batch_images[b].detach().cpu()
                        blended_img = ((blended_img + 1) / 2).clamp(0,1)              # -> [0,1]
                        blended_img = torchvision.transforms.ToPILImage()(blended_img)

                        orig_t = original_batch_tensors[b].detach().cpu()     # CHW, in [-1,1]
                        orig_t = ((orig_t + 1) / 2).clamp(0,1)              # -> [0,1]
                        Orig_img = torchvision.transforms.ToPILImage()(orig_t)
                        
                        axes[0].imshow(Orig_img); axes[0].set_title("Original")
                        axes[1].imshow(blended_img);  axes[1].set_title(f"k{k_blending}_Blended")
                        axes[2].imshow(generated_img);  axes[2].set_title("Generated")
                        plt.tight_layout()
                        plt.show()
                        generated_img.close()
              
        latents_save_path = os.path.join(Save_monitoring_path, 'all_latents_with_meta.pt')
        torch.save({
            "latents": torch.cat(all_latents, dim=0),
            "prompts": all_prompts,
            "image_names": all_image_names
        }, latents_save_path)
    # To Verify the File
    data = torch.load(latents_save_path)
    print(data['latents'].shape)
    print(data['prompts'][0])
    print(data['image_names'][0])