import torch
import os
from PIL import Image

class BiomassDataset(torch.utils.data.Dataset):
    def __init__(self, df, base_path, transform=None):
        self.df = df
        self.base_path = base_path
        self.transform = transform
        self.target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
        
    def __len__(self):
        return len(self.df) * 2  # Double the size for left/right splits
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx // 2]  # Map to original image
        img_path = os.path.join(self.base_path, row['image_path'])
        targets = row[self.target_cols].values.astype('float32')
        
        image = Image.open(img_path).convert('RGB')
        
        # Split left or right
        half = idx % 2
        width, height = image.size
        if half == 0:
            image = image.crop((0, 0, width // 2, height))  # Left half
        else:
            image = image.crop((width // 2, 0, width, height))  # Right half
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(targets, dtype=torch.float32)
