from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import ImageColor, Image, ImageDraw
import os


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

def load_and_transform_dataset(Image_size, image_path, text_path, tokenizer, textencoder, use_augmentation=True):
    if use_augmentation:
        data_transforms = transforms.Compose([
            transforms.Resize((Image_size, Image_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            ),
            transforms.RandomAffine(
                degrees=(-2, 2),
                translate=(0.02, 0.02),
                scale=(0.98, 1.02),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)  # Normalize to [-1, 1]
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.Resize((Image_size, Image_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)  # Normalize to [-1, 1]
        ])          
    return ConditionalImageDataset(image_path, text_path, tokenizer=tokenizer, textencoder=textencoder, transform=data_transforms), tokenizer, textencoder
