# %%
import os
import cv2
import time
import torch
import random
import argparse

import numpy as np
import torchvision
from tqdm import tqdm
from glob import glob
import mediapipe as mp
from PIL import ImageColor, Image, ImageDraw

import torch.nn.functional as F
from collections import OrderedDict
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel 
from transformers import CLIPTokenizer, CLIPTextModel

from vae import VAE
from unet_cond import Unet
from linear_noise_scheduler import LinearNoiseScheduler

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--text_weight', type=float, default=1.0, help='Weight for text conditioning (0.0-1.0)')
parser.add_argument('-std', '--privacy_noise_std', type=float, default=0.0, help='Latent noise injection')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
elif torch.xpu.is_available():
    import intel_extension_for_pytorch as ipex
    device = torch.device("xpu")
    print("XPU is available. Using XPU.")
else:
    device = torch.device("cpu")
    print("CUDA and XPU are not available. Using CPU.")

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clean_prompt(raw_prompt, wrap=False):
    """
    Cleans a raw prompt by removing commas and underscores, 
    lowercasing, and optionally wrapping in a descriptive phrase.

    Args:
        raw_prompt (str): e.g. "Asian_chinese_korean, Female, Young"
        wrap (bool): If True, wraps the prompt in a sentence.

    Returns:
        str: Cleaned and formatted prompt
    """
    raw_prompt = raw_prompt.lower()
    # print('raw_prompt', raw_prompt)
    # raw_prompt = raw_prompt.replace("male", "man").replace("female", "woman")
    # cleaned = raw_prompt.replace("_", " ").lower()
    cleaned = raw_prompt #" ".join(cleaned.split())  # remove extra spaces
    # print('raw_prompt', cleaned)
    return f"a portrait of a {cleaned}" if wrap else cleaned

class ConditionalImageDataset(Dataset):
    def __init__(self, img_dir, txt_dir, tokenizer, textencoder, transform=None):
        self.img_dir = img_dir
        self.txt_dir = txt_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.textencoder = textencoder        
        
    def __len__(self):
        return len(self.img_dir)# * self.repeats

    def __getitem__(self, idx):
        img_idx = idx #// self.repeats 
        img_path = self.img_dir[img_idx]        
        image = Image.open(img_path).convert("RGB")
        image_name = os.path.basename(img_path) # This just for the imge name
        with open(self.txt_dir[img_idx], "r", encoding="utf-8") as f:
            lines = [line.strip() for _, line in zip(range(1), f) if line.strip()]
        prompt = " ".join(lines)
        with open(self.txt_dir[img_idx], "r", encoding="utf-8") as f:
            prompt = f.read().strip()  

        image = self.transform(image)
        tokenized_text = self.tokenizer(prompt, padding="max_length", truncation=True, max_length=77, return_tensors="pt")#.to(device)
        text_original = prompt     
        indexed_tokens = tokenized_text['input_ids'].squeeze(0)
        att_mask = tokenized_text['attention_mask'].squeeze(0)
        
        # Text Normalization added to the model architecture                    
        return image, tokenized_text, text_original, image_name,  indexed_tokens, att_mask #, img_original  # Dummy label since no classes exist

def load_and_transform_dataset():
    data_transforms = transforms.Compose([
    transforms.Resize((Image_size, Image_size), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)  # Normalize to [-1, 1]
    ])          
    return ConditionalImageDataset(image_path, text_path, tokenizer=tokenizer, textencoder=textencoder, transform=data_transforms), tokenizer, textencoder

def get_face_pose_euler_angles(img):
    """
    Extracts yaw, pitch, and roll from a single face image tensor (C, H, W) in range [-1, 1]
    """
    # Convert torch tensor to uint8 image for MediaPipe
    if isinstance(img, torch.Tensor):
        img_np = (img.permute(1, 2, 0).cpu().numpy() + 1) / 2  # [-1,1] -> [0,1]
        img_np = (img_np * 255).astype(np.uint8)
    else:
        raise ValueError("Expected image as torch.Tensor in shape [C, H, W]")

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]

        # if len(face_landmarks.landmark) != 468:
        #     return None  # reject bad detection

        # Get selected 2D image points
        image_points = np.array([
            [face_landmarks.landmark[1].x * img_np.shape[1],  face_landmarks.landmark[1].y * img_np.shape[0]],    # Nose tip
            [face_landmarks.landmark[33].x * img_np.shape[1], face_landmarks.landmark[33].y * img_np.shape[0]],   # Left eye
            [face_landmarks.landmark[263].x * img_np.shape[1], face_landmarks.landmark[263].y * img_np.shape[0]], # Right eye
            [face_landmarks.landmark[61].x * img_np.shape[1], face_landmarks.landmark[61].y * img_np.shape[0]],   # Mouth left
            [face_landmarks.landmark[291].x * img_np.shape[1], face_landmarks.landmark[291].y * img_np.shape[0]], # Mouth right
            [face_landmarks.landmark[199].x * img_np.shape[1], face_landmarks.landmark[199].y * img_np.shape[0]], # Forehead
            [face_landmarks.landmark[152].x * img_np.shape[1], face_landmarks.landmark[152].y * img_np.shape[0]], # Chin
            ], dtype='double')

        # 3D model points (approximate values in mm)
        model_points = np.array([
            [0.0, 0.0, 0.0],        # Nose tip
            [-30.0, -30.0, -30.0],  # Left eye
            [30.0, -30.0, -30.0],   # Right eye
            [-30.0, 30.0, -30.0],   # Mouth left
            [30.0, 30.0, -30.0],    # Mouth right
            [0.0, 50.0, -30.0],     # Forehead
            [0.0, -63.0, -30.0]     # Chin
        ], dtype='double')

        focal_length = img_np.shape[1]
        center = (img_np.shape[1] / 2, img_np.shape[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype='double')

        dist_coeffs = np.zeros((4, 1))  # no distortion

        success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Convert rotation matrix to Euler angles
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            yaw = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            yaw = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = 0

        # Convert to degrees
        yaw = np.degrees(yaw)
        pitch = np.degrees(pitch)
        roll = np.degrees(roll)

        return [yaw, pitch, roll]
    
def organize_images_by_class(dataloader):
    """
    Organizes images by their demographic class based on prompts
    Returns a dictionary with class combinations as keys and lists of (image, prompt, img_name, person_id) tuples as values
    """
    class_images = {}
    
    for batch_idx, (images, _, prompts, img_names, _, _) in enumerate(dataloader):
        for img, prompt, img_name in zip(images, prompts, img_names):
            # Extract person ID from image name (assuming format: personID_*.*)
            person_id = os.path.splitext(os.path.basename(img_name))[0].split('_')[0]
            
            # Clean and standardize the prompt
            class_key = clean_prompt(prompt, wrap=False)
            pose = get_face_pose_euler_angles(img)  # Add this line
            if pose is None:
                pose = [0.0, 0.0, 0.0]
                # continue  # Skip if pose is not detected
            if class_key not in class_images:
                class_images[class_key] = []
            class_images[class_key].append((img, prompt, img_name, person_id, pose))
    
    return class_images

# Distance function
def pose_distance(p1, p2):
    return sum(abs(a - b) for a, b in zip(p1, p2))

def sample_with_img(model, scheduler, vae, text_weight=None):
    model.eval()
    all_latents = []
    all_prompts = []
    all_image_names = []
    # Organize images by class first
    class_images = organize_images_by_class(train_dataloader)
    im_size = Image_size // 2 ** sum([True, True])
    start_batch = 0  # set your desired starting batch index
    if start_batch > 0:
        print('Save_monitoring_path::::', Save_monitoring_path)
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

            # def pose_distance(p1, p2):
            #     return np.linalg.norm(np.array(p1) - np.array(p2))  # Euclidean distance
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
                    # one_sim = one_sim + 1
                    # Sort all available other-person images by pose
                    similar_images = sorted(other_people_images, key=lambda x: pose_distance(x[-1], current_pose))

                    # === Uniform averaging with original ===
                    imgs_stack = torch.stack([img_data[0].float() for img_data in similar_images])
                    original_img_tensor = current_img_data[0][0].float()
                    imgs_stack_with_original = torch.cat([imgs_stack, original_img_tensor.unsqueeze(0)], dim=0)
                    merged_img = imgs_stack_with_original.mean(dim=0).clamp(-1, 1)

                    print(f"[✓] Using less images {len(similar_images)} than k_{k_blending} from another person for class {class_key}")
                else:
                    # same_sim = same_sim + 1
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
        for i in tqdm(step_indices, desc="Denoising with fewer steps"):
            i = i.item()

            if text_embeddings.shape[0] != img_hidden_states.shape[0]:
                min_batch = min(text_embeddings.shape[0], img_hidden_states.shape[0])
                text_embeddings = text_embeddings[:min_batch]
                img_hidden_states = img_hidden_states[:min_batch]
                xt = xt[:min_batch]
                batch_img_names = batch_img_names[:min_batch]
                txt_original = txt_original[:min_batch]
                print(f"⚠️ Batch size mismatch: text={text_embeddings.shape[0]}, image={img_hidden_states.shape[0]}. Truncating to {min_batch}.")                
                                            
            noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device), text_embeddings.to(device), img_hidden_states, text_weight, image_weight)
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

            if i == 0:
                all_latents.append(xt.cpu())
                all_prompts.extend(txt_original)
                all_image_names.extend(batch_img_names)

                privacy_noise_std = args.privacy_noise_std
                if privacy_noise_std > 0:
                    noise = torch.randn_like(xt) * privacy_noise_std
                    xt = xt + noise
                                
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
                    img = torchvision.transforms.ToPILImage()(ims[6])
                    img.save(os.path.join(dataset_name, f'samples_img_txt_{time_steps}', 'x0_{}.png'.format(i)))
                # img = torchvision.transforms.ToPILImage()(grid[0])
                # img.save(os.path.join(dataset_name, f'samples_img_txt_{time_steps}', 'x0_{}.png'.format(i)))
                if i == 0:                                                     
                    for b, img_name in enumerate(batch_img_names):
                        single_img = torchvision.transforms.ToPILImage()(ims[b])
                        original_name = img_name
                        new_name = os.path.splitext(original_name)[0] + "_gen"
                        person_id = os.path.splitext(os.path.basename(original_name))[0].split('_')[0]
                        person_folder = os.path.join(Save_monitoring_path, 'Generated_images', person_id)
                        os.makedirs(person_folder, exist_ok=True)
                        single_img.save(os.path.join(person_folder, f'{new_name}_{i}.png'))
                        single_img.save(os.path.join(Save_monitoring_path, f'{new_name}_{i}.png'))
                        single_img.close()
              
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
    
def main():
    with torch.no_grad():
        sample_with_img(model, scheduler, vae, args.text_weight) #text_weight = 1.0) #args.text_weight)

if __name__ == "__main__":
    Imgweights = [0.700] #[0.850, 0.700, 0.550, 0.250, 0.950, 0.400] #[0.700, 0.550, 0.250, 0.950, 0.850, 0.400]
    dir_path = os.path.dirname(__file__)
    Diff_model = './models' #'Celebahq_MTF_4_64_64_Batch64_Latent_ImgTextTraining' #'Celebahq_MTF_4_64_64_Batch64_Latent_Transfer_toMTF_ManWoman_ImgTextTraining' #'Celebhq_ShortPrompt_4_64_64_Batch64_Latent_DecreaseWeight'
    VAE_model = './models' #'Celebahq_MTF_4_64_64_Batch64' #'Celebahq_MTF_4_64_64_Batch64' 
    dataset_name = 'MTF_generation' # Create the noise scheduler Testing_500_steps_poseAveragingWithOriginal_ImgTxtWeighted
    Image_size = 128
    batch_size = 8
    time_steps = 500 # put the same value for the time_steps and num_steps
    num_steps = time_steps  # set this to your desired number of steps for inferencing
    Data_dir = "C:/M_ali/LDM_Rami/Final_LDM" 
    # # path normal data without anonymization
    MTF_Image_data_dir = Data_dir + "/Data/test/image/"
    MTF_Text_data_dir = Data_dir + "/Data/test/prompt/"

    print('Data path: ', MTF_Text_data_dir)
    scale_factor = 1 #1.0027 #0.92891 #1.12001# 0.92891 #0.92891 # 0.18215 #1.0026573259687683 - std of Celba+MT

    for imgw in Imgweights:
        for xy in range(3, 4):
            k_blending = xy # The value of blending or class-based factor (2-class, 3-class, 4-class, etc.)
            print(f'k-blending: {k_blending}, args.text_weight: {imgw}') #args.text_weight)
            args.text_weight = imgw
            image_weight = 1.0 - args.text_weight 
            
            # Path of save Testing 
            Save_monitoring_path = os.path.join(dir_path, dataset_name, f"Testing_{time_steps}_steps_text_image/Class-based_Merging_K{k_blending}/PoseSim/MTF128_imgW_{np.round(image_weight,3)}_std={args.privacy_noise_std}")
        
            if not os.path.exists(Save_monitoring_path):
                if not os.path.exists(os.path.join(dir_path, dataset_name)):
                    os.makedirs(os.path.join(dir_path, dataset_name), exist_ok=True)
                os.makedirs(Save_monitoring_path, exist_ok=True)
            image_path = sorted(glob(os.path.join(MTF_Image_data_dir, '*.*')))
            text_path = sorted(glob(os.path.join(MTF_Text_data_dir, '*.*')))

            image_path = sorted(list(image_path))
            text_path = sorted(list(text_path))

            print(f"Total samples of images: {len(image_path)}, Text samples: {len(text_path)}")
            N_train = len(image_path)
            print('image_path: ', image_path[:2])
            print('text_path: ',text_path[:2])
            clip_model_name = "openai/clip-vit-large-patch14"
            # if local_rank == 0:
            tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
            processor = CLIPProcessor.from_pretrained(clip_model_name)

            textencoder = CLIPTextModel.from_pretrained(clip_model_name).to(device) 
            clip_model = CLIPModel.from_pretrained(clip_model_name, output_hidden_states=True).to(device) 

            # Optional: Freeze CLIP weights if you don't plan to fine-tune it
            for param in clip_model.parameters():
                param.requires_grad = False
            for param in textencoder.parameters():
                param.requires_grad = False
            # if local_rank == 0:
            print("Loaded CLIP models successfully!")
            # Load dataset
            train_dataset, tokenizer, textencoder = load_and_transform_dataset()
            print('train_dataset: ', len(train_dataset))  # Check the number of images
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

            vae = VAE(im_channels=3).to(device)  
            # vae = vae.float()
            model = Unet(im_channels=4).to(device)

            scheduler = LinearNoiseScheduler(num_timesteps=time_steps, #diffusion_config['num_timesteps'],
                                        beta_start=1e-4, #diffusion_config['beta_start'],
                                        beta_end=0.02, #diffusion_config['beta_end'],
                                        ldm_scheduler=True)

            # Load the pre-traiened the VAE model
            if os.path.exists(os.path.join(dir_path, VAE_model,'vae_autoencoder.pth')):        
                vae.load_state_dict(torch.load(os.path.join(dir_path, VAE_model, 'vae_autoencoder.pth'), map_location=device))
                vae.eval()                   
                vae.float()                  
                for param in vae.parameters():
                    param.requires_grad = False                
                print('--Loaded vae checkpoint--')
            else:
                print("There is no checkpoint of vae loaded")

            # Model 500 steps based
            if os.path.exists(os.path.join(dir_path, Diff_model, 'ddpm_500.pth')):
                checkpoint = torch.load(os.path.join(dir_path, Diff_model, 'ddpm_500.pth'), map_location='cpu')
                state_dict = checkpoint['model_state_dict']

                # Remove 'module.' prefix if needed
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_key = k.replace("module.", "")  # remove 'module.' prefix
                    new_state_dict[new_key] = v

                model.load_state_dict(new_state_dict)
                print("Diffusion model loaded successfully")
            else:
                print("There is no checkpoint of diffusion model loaded")
            main()