import os
import json
from PIL import Image
from glob import glob

from torch.utils.data import Dataset

class Cifar100Dataset(Dataset):
    def __init__(self, data_path, type_, transform=None):
        super(Cifar100Dataset, self).__init__()
        self.transform = transform
        self.image_paths = glob(os.path.join(data_path, f"{type_}_images", "*", "*", "*"))
        with open(os.path.join(data_path, "label_map.json"), 'r') as f:
            self.label_map = json.load(f)
        self.fine_map = []
        self._make_fine_map()
        
    def _make_fine_map(self):
        for key, values in self.label_map.items():
            self.fine_map.extend([f"{key}-{value}" for value in values])
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index, transform=None):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert("RGB")
        
        fine_label = img_path.split('/')[-2]
        coarse_label = img_path.split('/')[-3]
        fine_label = self.fine_map.index(f"{coarse_label}-{fine_label}")
        coarse_label = list(self.label_map.keys()).index(coarse_label)
        
        if self.transform:
            img = self.transform(img)
        
        return img, coarse_label, fine_label