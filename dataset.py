import os
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
import torchvision.transforms.functional as TF

SEMANTIC_KEYS_ORDER = [
    "morphology",               
    "margin_definition",        
    "internal_texture",         
    "surrounding_interaction",  
    "boundary_distinctness",    
    "malignancy_risk",          
    "pathological_inference",  
    "differential_reasoning",  
    "predicted_diagnosis"      
]

class SemanticSegmentationDataset(data.Dataset):

    def __init__(self, image_root, gt_root, trainsize, props_root_name="properties_cache_384", augment=True):
        self.trainsize = trainsize
        self.props_root_name = props_root_name 
        self.augment = augment
        
        file_write_obj_img = image_root
        file_write_obj_gt = gt_root

        self.img_list = []
        self.gt_list = []
        self.props_list = []

        with open(file_write_obj_img, "r") as imgs:
            for img_path in imgs:
                img_path = img_path.rstrip('\n')
                if not img_path: continue
                self.img_list.append(img_path)
                
                img_dir = os.path.dirname(img_path)
                dataset_root = os.path.dirname(img_dir) 
                props_dir = os.path.join(dataset_root, self.props_root_name)
                img_filename = os.path.basename(img_path)
                props_filename = os.path.splitext(img_filename)[0] + ".pt"
                props_path = os.path.join(props_dir, props_filename)
                self.props_list.append(props_path)

        with open(file_write_obj_gt, "r") as gts:
            for gt in gts:
                _video = gt.rstrip('\n')
                if not _video: continue
                self.gt_list.append(_video)

        self.img_list = sorted(self.img_list)
        self.gt_list = sorted(self.gt_list)
        self.props_list = sorted(self.props_list)

        if len(self.img_list) != len(self.gt_list) or len(self.img_list) != len(self.props_list):
            raise ValueError(f"数据集列表长度不匹配: "
                             f"Images ({len(self.img_list)}), "
                             f"GTs ({len(self.gt_list)}), "
                             f"Props ({len(self.props_list)})")

        self.size = len(self.img_list)
        
        self.resize_img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.resize_gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        
        self.normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __getitem__(self, index):
        img_path = self.img_list[index]
        gt_path = self.gt_list[index]
        props_path = self.props_list[index]
        
        image_pil = self.rgb_loader(img_path)
        gt_pil = self.binary_loader(gt_path)
        
        image = self.resize_img_transform(image_pil) 
        gt = self.resize_gt_transform(gt_pil)       
        
        gt = (gt != 0).float()

        is_hflipped = False
        is_vflipped = False
        is_rotated = False

        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                gt = TF.hflip(gt)
                is_hflipped = True

            if random.random() > 0.5:
                image = TF.vflip(image)
                gt = TF.vflip(gt)
                is_vflipped = True
            
            if random.random() > 0.8:
                angle = random.randint(-10, 10)
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                gt = TF.rotate(gt, angle, interpolation=TF.InterpolationMode.NEAREST)
                gt = (gt != 0).float() 
                is_rotated = True

            if random.random() > 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                contrast_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness_factor)
                image = TF.adjust_contrast(image, contrast_factor)
            
            if random.random() > 0.8:
                noise = torch.randn_like(image) * 0.05 
                image = image + noise
                image = torch.clamp(image, 0, 1)

        image = self.normalize_transform(image)
        
        properties_raw = torch.load(props_path, map_location='cpu')
        properties = {}
        
        for k, v in properties_raw.items():
            if not isinstance(v, torch.Tensor):
                tensor_v = torch.tensor(v, dtype=torch.float32)
            else:
                tensor_v = v.float()
            
            if tensor_v.dim() == 0:
                tensor_v = tensor_v.unsqueeze(0)
            
            if is_hflipped:
                if k == 'bbox':
                    old_xmin, old_xmax = tensor_v[0].clone(), tensor_v[2].clone()
                    tensor_v[0] = 1.0 - old_xmax
                    tensor_v[2] = 1.0 - old_xmin
                elif k == 'centroid':
                    tensor_v[0] = 1.0 - tensor_v[0]
                elif k == 'orientation':
                    tensor_v = 1.0 - tensor_v

            if is_vflipped:
                if k == 'bbox':
                    old_ymin, old_ymax = tensor_v[1].clone(), tensor_v[3].clone()
                    tensor_v[1] = 1.0 - old_ymax
                    tensor_v[3] = 1.0 - old_ymin
                elif k == 'centroid':
                    tensor_v[1] = 1.0 - tensor_v[1]
                elif k == 'orientation':
                    tensor_v = 1.0 - tensor_v

            if is_rotated:
                if k in ['bbox', 'centroid', 'orientation']:
                    tensor_v = torch.ones_like(tensor_v) * -1.0
            
            properties[k] = tensor_v

        img_dir = os.path.dirname(img_path)       
        dataset_root = os.path.dirname(img_dir)   
        embed_dir = os.path.join(dataset_root, "vlm_embeddings_gemini")
        
        img_filename = os.path.basename(img_path)
        embed_filename = os.path.splitext(img_filename)[0] + ".npy"
        embed_path = os.path.join(embed_dir, embed_filename)

        semantic_tensor = self.load_semantic_embedding(embed_path)

        dataset_id = self.get_dataset_id_from_path(img_path)

        return image, gt, properties, semantic_tensor, dataset_id

    def load_semantic_embedding(self, path):
            
        try:
            data_dict = np.load(path, allow_pickle=True).item()
            vectors = []
            for key in SEMANTIC_KEYS_ORDER:
                if key in data_dict:
                    vec = data_dict[key]
                    if vec.ndim == 1: vec = vec[np.newaxis, :]
                    vectors.append(vec)
                else:
            
                    vectors.append(np.zeros((1, 768), dtype=np.float32))
            
            stacked = np.concatenate(vectors, axis=0)
            return torch.from_numpy(stacked).float()
            
        except Exception as e:
            raise RuntimeError(f"")


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = self.resize_if_needed(img, False)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = self.resize_if_needed(img, True)
            return img.convert('L')
            
    def resize_if_needed(self, img, is_mask=True):
        w, h = img.size
        if max(w, h) > 1300:
            scale = 1300 / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            if is_mask:
                img = img.resize((new_w, new_h), Image.NEAREST)
            else:
                img = img.resize((new_w, new_h), Image.BILINEAR)
        return img

    def __len__(self):
        return self.size

class SemanticSegmentationTestDataset(data.Dataset):

    def __init__(self, image_list_file, gt_list_file, trainsize, props_root_name=None):
        self.trainsize = trainsize
        self.props_root_name = props_root_name or f"properties_cache_{trainsize}"

        with open(image_list_file, 'r') as f:
            self.img_list = [line.strip() for line in f.readlines() if line.strip()]
        with open(gt_list_file, 'r') as f:
            self.gt_list = [line.strip() for line in f.readlines() if line.strip()]

        if len(self.img_list) != len(self.gt_list):
            raise ValueError(f"")

        self.img_list = sorted(self.img_list)
        self.gt_list = sorted(self.gt_list)

        self.resize_img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.resize_gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        self.normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.img_list)

    def _props_path(self, img_path):
        img_dir = os.path.dirname(img_path)
        dataset_root = os.path.dirname(img_dir)
        props_dir = os.path.join(dataset_root, self.props_root_name)
        props_filename = os.path.splitext(os.path.basename(img_path))[0] + '.pt'
        return os.path.join(props_dir, props_filename)

    def _embed_path(self, img_path):
        img_dir = os.path.dirname(img_path) 
        dataset_root = os.path.dirname(img_dir)
        embed_dir = os.path.join(dataset_root, "vlm_embeddings_gemini")
        embed_filename = os.path.splitext(os.path.basename(img_path))[0] + ".npy"
        return os.path.join(embed_dir, embed_filename)

    def load_semantic_embedding(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"")
            
        try:
            data_dict = np.load(path, allow_pickle=True).item()
            vectors = []
            for key in SEMANTIC_KEYS_ORDER:
                if key in data_dict:
                    vec = data_dict[key]
                    if vec.ndim == 1: vec = vec[np.newaxis, :]
                    vectors.append(vec)
                else:
                    vectors.append(np.zeros((1, 768), dtype=np.float32))
            stacked = np.concatenate(vectors, axis=0)
            return torch.from_numpy(stacked).float()
        except Exception as e:
            raise RuntimeError(f"")


    def __getitem__(self, index):
        img_path = self.img_list[index]
        gt_path = self.gt_list[index]
        
        image = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        w_, h_ = image.size

        image_tensor = self.resize_img_transform(image)
        image_tensor = self.normalize_transform(image_tensor)
        gt_tensor = self.resize_gt_transform(gt)
        gt_tensor = (gt_tensor != 0).float()

        return image_tensor, gt_tensor, w_, h_, img_path

    def __init__(self, image_list_file, gt_list_file, trainsize, props_root_name=None):
        self.trainsize = trainsize
        self.props_root_name = props_root_name or f"properties_cache_{trainsize}"

        with open(image_list_file, 'r') as f:
            self.img_list = [line.strip() for line in f.readlines() if line.strip()]
        with open(gt_list_file, 'r') as f:
            self.gt_list = [line.strip() for line in f.readlines() if line.strip()]

        if len(self.img_list) != len(self.gt_list):
            raise ValueError(f"")

        self.img_list = sorted(self.img_list)
        self.gt_list = sorted(self.gt_list)

        self.resize_img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.resize_gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        self.normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.img_list)

    def _props_path(self, img_path):
        img_dir = os.path.dirname(img_path)
        dataset_root = os.path.dirname(img_dir)
        props_dir = os.path.join(dataset_root, self.props_root_name)
        props_filename = os.path.splitext(os.path.basename(img_path))[0] + '.pt'
        return os.path.join(props_dir, props_filename)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        gt_path = self.gt_list[index]
        image = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        w_, h_ = image.size

        image_tensor = self.resize_img_transform(image)
        image_tensor = self.normalize_transform(image_tensor)
        gt_tensor = self.resize_gt_transform(gt)
        gt_tensor = (gt_tensor != 0).float() 

        props_path = self._props_path(img_path)
        props_raw = torch.load(props_path, map_location='cpu')
        props = {}
        for k, v in props_raw.items():
            if not isinstance(v, torch.Tensor):
                tensor_v = torch.tensor(v, dtype=torch.float32)
            else:
                tensor_v = v.float()

            if tensor_v.dim() == 0:
                tensor_v = tensor_v.unsqueeze(0)
            elif tensor_v.dim() == 1:

                pass  
            
            props[k] = tensor_v

        name = os.path.basename(img_path)
        return image_tensor, gt_tensor, props, name, w_, h_


def get_loader_semantic(image_root, gt_root, batchsize, trainsize, distributed=True, augment=True):

    dataset = SemanticSegmentationDataset(image_root, gt_root, trainsize, augment=augment)
    
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    if distributed:
        num_workers = 12 
        pin_memory = True  
        prefetch_factor = 2  
    else:
        num_workers = 12  
        pin_memory = True
        prefetch_factor = 2
    
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  sampler=sampler,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn,
                                  prefetch_factor=prefetch_factor)

    return data_loader, sampler

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)