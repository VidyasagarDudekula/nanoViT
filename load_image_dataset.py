import torchvision
from datasets import load_dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import torch
from torch.utils.data import Dataset

def load_data(mode='train'):
    tiny_imagenet = load_dataset('Maysee/tiny-imagenet', split=mode)
    return tiny_imagenet

class ImageDataset(Dataset):
    def __init__(self, config):
        self.mode = config.mode
        self.image_size = config.image_size
        self.ds = load_data(self.mode)
        self.trans_module = None
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        if self.mode:
            self.trans_module = transforms.Compose([
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.ToTensor(),
                transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR),
                normalize,
            ])
        else:
            self.trans_module = transforms.Compose([
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.ToTensor(),
                transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR),
                normalize,
            ])
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        indexed_data = self.ds[idx]
        img_obj = indexed_data['image']
        label = indexed_data['label']
        label_tensor = torch.tensor(label, dtype=torch.long).view(-1)
        img_obj = self.trans_module(img_obj)
        return img_obj, label_tensor

if __name__ == '__main__':
    from config import ModelArgs
    config = ModelArgs()
    m = ImageDataset(config)
    import pdb; pdb.set_trace()
    x = m[0]
