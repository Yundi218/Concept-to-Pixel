import argparse
import os
import torch
import torch.nn.functional as F
from network import SRICL
from dataset import get_loader_semantic
from utils import AvgMeter
from dataset import SemanticSegmentationTestDataset
import torch.nn as nn
from torch.cuda import amp
import torch.distributed as dist
from contextlib import contextmanager
import torch.utils.data as data
import numpy as np
import random
from itertools import cycle
from tqdm import tqdm

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def structure_loss_improved(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    pred_prob = torch.sigmoid(pred)

    p_t = (mask * pred_prob) + ((1 - mask) * (1 - pred_prob))
    focal_weight = (1 - p_t) ** 2.0 

    wbce = (weit * focal_weight * bce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + 1e-8)
    
    inter = ((pred_prob * mask) * weit).sum(dim=(2, 3))
    union = ((pred_prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    
    return (wbce + wiou).mean()



def dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    dice_coeff = (2. * intersection + smooth) / \
        (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
    dice_loss = 1 - dice_coeff.mean()
    return dice_loss


def calculate_iou(pred_mask, target):
    # pred_mask 已经是概率图或二值图
    pred = (pred_mask > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - intersection
    iou = (intersection + 1e-5) / (union + 1e-5)
    return iou.mean().item()

def calculate_dice(pred_mask, target):
    pred = (pred_mask > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3))
    dice = (2. * intersection + 1e-5) / \
        (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-5)
    return dice.mean().item()

def compute_mask_props(mask_prob):
    device = mask_prob.device
    H, W = mask_prob.shape
    area_pixels = mask_prob.sum()
    area_norm = area_pixels / (H * W)
    y_coords = torch.arange(H, device=device).float().unsqueeze(1).expand(H, W)
    x_coords = torch.arange(W, device=device).float().unsqueeze(0).expand(H, W)
    denom = area_pixels + 1e-6
    cy = (y_coords * mask_prob).sum() / denom / H 
    cx = (x_coords * mask_prob).sum() / denom / W 
    return area_norm, torch.tensor([cx, cy], device=device)

def predict_with_token_tta_fast(model, image, h_orig, w_orig, use_tta=True):
    model.eval()
    
    if not use_tta:
        with torch.no_grad():
            with amp.autocast(enabled=True):
                output_prior, _, _, _ = model(image)
            mask_pred = F.interpolate(output_prior, size=(h_orig, w_orig), mode='bilinear', align_corners=True)
            return torch.sigmoid(mask_pred)

    img_batch = torch.cat([
        image,
        torch.flip(image, [3]),
        torch.flip(image, [2])
    ], dim=0) 
    with torch.no_grad():
        with amp.autocast(enabled=True):
            output_prior_batch, _, geo_pred_batch, _ = model(img_batch)

    output_prior_batch = output_prior_batch.float()
    mask_preds_resized = F.interpolate(output_prior_batch, size=(h_orig, w_orig), mode='bilinear', align_corners=True)
    mask_probs_batch = torch.sigmoid(mask_preds_resized) 
    
    final_mask_accum = torch.zeros((1, 1, h_orig, w_orig), device=image.device)
    total_weight = 0.0
    token_areas = []
    
    inv_funcs = [lambda m: m, lambda m: torch.flip(m, [3]), lambda m: torch.flip(m, [2])]

    for i in range(3):
        mask_prob = mask_probs_batch[i:i+1]
        
        if geo_pred_batch is not None and 'area' in geo_pred_batch:
            pred_area_token = geo_pred_batch['area'][i].item()
            token_areas.append(pred_area_token)
            pred_centroid_token = geo_pred_batch['centroid'][i] if 'centroid' in geo_pred_batch else None
        else:
            pred_area_token = None
            token_areas.append(1.0)

        confidence_weight = 1.0
        if pred_area_token is not None:
            mask_area, mask_cent = compute_mask_props(mask_prob.squeeze(0).squeeze(0))
            diff_area = abs(mask_area.item() - pred_area_token)
            diff_cent = torch.norm(mask_cent - pred_centroid_token).item() if pred_centroid_token is not None else 0.0
            inconsistency = 10.0 * diff_area + 1.0 * diff_cent
            confidence_weight = torch.exp(torch.tensor(-inconsistency * 5.0)).item()

        final_mask_accum += inv_funcs[i](mask_prob) * confidence_weight
        total_weight += confidence_weight

    final_mask = final_mask_accum / (total_weight + 1e-8)
    
    avg_token_area = sum(token_areas) / len(token_areas)
    if avg_token_area < 0.001 and final_mask.mean() < 0.005:
        return torch.zeros_like(final_mask)

    return final_mask


def is_dist_avail_and_initialized():
    if not dist.is_available(): return False
    if not dist.is_initialized(): return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized(): return 0
    return dist.get_rank()

def is_master():
    return get_rank() == 0

@contextmanager
def torch_distributed_zero_first(rank: int):
    if is_dist_avail_and_initialized() and rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if is_dist_avail_and_initialized() and rank == 0:
        torch.distributed.barrier()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help='epoch number')
    parser.add_argument('--lr_gen', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
    parser.add_argument('-beta1_gen', type=float, default=0.5, help='beta of Adam for generator')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight_decay')
    parser.add_argument('--feat_channel', type=int, default=64, help='reduced channel of saliency feat')
    parser.add_argument('--save_path', type=str, default='./saved_model/',
                        help='path to save model checkpoints and logs')
    parser.add_argument('--eval_interval', type=int, default=1, help='evaluate every N epochs')
    parser.add_argument('--save_interval', type=int, default=10, help='save checkpoint every N epochs')
    parser.add_argument('--save_start_epoch', type=int, default=40, help='start saving checkpoints after this epoch')
    return parser.parse_args()


def train():
    opt = get_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available(): torch.cuda.set_device(local_rank)
    if distributed:
        dist.init_process_group(backend="nccl", world_size=world_size, init_method="env://")

    is_master_node = (distributed and (local_rank == 0)) or (not distributed)
    seed_torch(42)

    data_lists_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_lists')
    train_datasets_config = [
        ('amdsd', 'amdsd_train_images.txt', 'amdsd_train_masks.txt'),
        ('btd', 'btd_train_images.txt', 'btd_train_masks.txt'),
        ('ebhi', 'ebhi_train_images.txt', 'ebhi_train_masks.txt'),
        ('tnui', 'tnui_train_images.txt', 'tnui_train_masks.txt'),
        ('polyp', 'polyp_train_images.txt', 'polyp_train_masks.txt'),
        ('covid', 'covid_train_images.txt', 'covid_train_masks.txt'),
        ('breast', 'breast_train_images.txt', 'breast_train_masks.txt'),
        ('skin', 'isic2018_train_images.txt', 'isic2018_train_masks.txt'),
    ]
    loaders = []
    samplers = []
    loader_lengths = []
    
    for name, img_file, mask_file in train_datasets_config:
        img_root = os.path.join(data_lists_dir, img_file)
        gt_root = os.path.join(data_lists_dir, mask_file)
        loader, sampler = get_loader_semantic(
            img_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, distributed=distributed)
        loaders.append(loader)
        samplers.append(sampler)
        loader_lengths.append(len(loader))

    max_steps = max(loader_lengths)
    
    val_loaders = {}
    if is_master_node:
        val_configs = [
            {'name': 'AMDSD', 'test_img': 'amdsd_test_images.txt', 'test_gt': 'amdsd_test_masks.txt'},
            {'name': 'BTD', 'test_img': 'btd_test_images.txt', 'test_gt': 'btd_test_masks.txt'},
            {'name': 'EBHI', 'test_img': 'ebhi_test_images.txt', 'test_gt': 'ebhi_test_masks.txt'},
            {'name': 'TNUI', 'test_img': 'tnui_test_images.txt', 'test_gt': 'tnui_test_masks.txt'},
            {'name': 'Polyp', 'test_img': 'polyp_test_images.txt', 'test_gt': 'polyp_test_masks.txt'},
            {'name': 'COVID', 'test_img': 'covid_test_images.txt', 'test_gt': 'covid_test_masks.txt'},
            {'name': 'Breast', 'test_img': 'breast_test_images.txt', 'test_gt': 'breast_test_masks.txt'},
            {'name': 'ISIC2018', 'test_img': 'isic2018_test_images.txt', 'test_gt': 'isic2018_test_masks.txt'},
        ]
        for conf in val_configs:
            img_f = os.path.join(data_lists_dir, conf['test_img'])
            gt_f = os.path.join(data_lists_dir, conf['test_gt'])
            if os.path.exists(img_f) and os.path.exists(gt_f):
                ds = SemanticSegmentationTestDataset(img_f, gt_f, opt.trainsize, props_root_name=f"properties_cache_{opt.trainsize}")
                val_loaders[conf['name']] = data.DataLoader(
                    ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    size_rates = [1] 
    use_fp16 = True
    save_path = opt.save_path
    if not save_path.endswith('/'): save_path += '/'
    with torch_distributed_zero_first(rank=local_rank):
        os.makedirs(save_path, exist_ok=True)
    generator = SRICL(model='base') 
    generator.cuda()
    start_epoch = 1
    generator_optimizer = torch.optim.Adam(generator.parameters(), opt.lr_gen)

    scaler = amp.GradScaler(enabled=use_fp16)
    
    if distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    loss_sem_func = nn.CosineEmbeddingLoss(margin=0.0).cuda()
    for epoch in range(start_epoch, opt.epoch + 1):
        if distributed:
            for sampler in samplers: sampler.set_epoch(epoch)
        generator.train()
        loss_record = AvgMeter()
        cycled_loaders = [cycle(loader) if len(loader) < max_steps else loader for loader in loaders]
        
        if is_master_node:
            pbar = tqdm(zip(*cycled_loaders), total=max_steps, desc=f'Epoch [{epoch}/{opt.epoch}]', ncols=180)
        else:
            pbar = zip(*cycled_loaders)

        for i, batch_data in enumerate(pbar, start=1):
            images = [item[0].cuda(non_blocking=True) for item in batch_data]
            gts = [item[1].cuda(non_blocking=True) for item in batch_data]
            props = []
            for item in batch_data:
                props.append({k: v.cuda(non_blocking=True) for k, v in item[2].items()})
            semantic_gts = [item[3].cuda(non_blocking=True) for item in batch_data] 
            num_tasks = len(images)
            accumulated_loss = torch.tensor(0.0, device=images[0].device)
            accumulated_seg_loss = torch.tensor(0.0, device=images[0].device)
            accumulated_geo_loss = torch.tensor(0.0, device=images[0].device)
            accumulated_sem_loss = torch.tensor(0.0, device=images[0].device)
            
            for rate in size_rates:
                avg_grads = [torch.zeros_like(p) for p in generator.parameters()]
                for task_idx in range(num_tasks):
                    query_image = images[task_idx]
                    query_gt = gts[task_idx]
                    query_props = props[task_idx]
                    query_sem_gt = semantic_gts[task_idx] 
                
                    trainsize = int(round(opt.trainsize * rate / 32) * 32)
                    if rate != 1:
                        query_image = F.interpolate(query_image, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                        query_gt = F.interpolate(query_gt, size=(trainsize, trainsize), mode='nearest')
                    
                    with amp.autocast(enabled=use_fp16):
                        query_gt = (query_gt != 0).float()
                        output_prior, output_priorb, geo_pred, sem_pred = generator(query_image)

                        seg_loss = structure_loss_improved(output_prior, query_gt) \
                                    + structure_loss_improved(output_priorb, 1. - query_gt) \
                                    + dice_loss(output_prior, query_gt) \
                                    + dice_loss(output_priorb, 1. - query_gt)                                                        
                        
                        geo_loss = torch.tensor(0.0, device=query_gt.device)
                        if geo_pred is not None:
                             geo_keys = [('bbox', True), ('area', False), ('perimeter', False), 
                                         ('aspect_ratio', True), ('compactness', False), 
                                         ('centroid', True), ('eccentricity', True), 
                                         ('orientation', True), ('solidity', False)]
                             for key, is_dep in geo_keys:
                                 if key in geo_pred and key in query_props:
                                     pred = geo_pred[key]
                                     target = query_props[key]
                                     if is_dep:
                                         if target.dim() > 1: valid_mask = (target[:, 0] != -1)
                                         else: valid_mask = (target != -1)
                                         if valid_mask.sum() > 0:
                                            p_valid = pred[valid_mask]
                                            t_valid = target[valid_mask]
                                            geo_loss += F.mse_loss(p_valid, t_valid)
                                     else:
                                         geo_loss += F.mse_loss(pred, target)
                        
                        sem_loss = torch.tensor(0.0, device=query_gt.device)
                        if sem_pred is not None:

                            pred_flat = sem_pred.view(-1, 768)
                            target_flat = query_sem_gt.view(-1, 768)
                            target_ones = torch.ones(pred_flat.shape[0], device=query_gt.device)
                            sem_loss = loss_sem_func(pred_flat, target_flat, target_ones)

                        task_loss = 1.0 * seg_loss + 1.0 * geo_loss + 1.0 * sem_loss
                        
                    generator_optimizer.zero_grad()
                    scaler.scale(task_loss).backward()
                    for avg_grad, p in zip(avg_grads, generator.parameters()):
                        if p.grad is not None: avg_grad.add_(p.grad)
                    accumulated_loss += task_loss.detach()
                    accumulated_seg_loss += seg_loss.detach()
                    accumulated_geo_loss += geo_loss.detach()
                    accumulated_sem_loss += sem_loss.detach()

                for param, avg_grad in zip(generator.parameters(), avg_grads):
                    if param.grad is not None:
                        param.grad = (avg_grad / (num_tasks * len(size_rates))).to(param.dtype)
                scaler.step(generator_optimizer)
                scaler.update()

            mean_loss = accumulated_loss / (num_tasks * len(size_rates))
            

if __name__ == '__main__':
    train()