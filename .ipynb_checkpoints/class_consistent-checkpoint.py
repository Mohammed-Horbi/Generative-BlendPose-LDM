import os

from pose import get_face_pose_euler_angles


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

def organize_images_by_class(dataloader):
    """
    Organizes images by their demographic class based on prompts
    Returns a dictionary with class combinations as keys and lists of (image, prompt, img_name, person_id) tuples as values
    """
    class_images = {}
    print('--Class pool of images preparation--')
    print('--Poses calculation--')
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