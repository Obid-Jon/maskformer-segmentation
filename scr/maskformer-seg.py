import os
import random
import zipfile
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from sklearn.model_selection import train_test_split

from transformers import (
    MaskFormerConfig,
    MaskFormerImageProcessor,
    MaskFormerForInstanceSegmentation,
)

warnings.filterwarnings('ignore')


# Загрузка датасета

import gdown

file_id = "1elmjRAlJNoY5eMPQTmg2V-cnab6XEoxK"
ds_name = "files.zip"

gdown.download(id=file_id, output=ds_name, quiet=False)

with zipfile.ZipFile(ds_name, 'r') as zip_ref:
    zip_ref.extractall('/content/')

print("Датасет загружен")

# параметры модели 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Устройство: {device}")

batch_size = 4
num_epochs = 8
learning_rate = 5e-5
im_size = 512

model_name = "facebook/maskformer-swin-base-coco"
model = MaskFormerForInstanceSegmentation.from_pretrained(model_name).to(device)
config = MaskFormerConfig.from_pretrained(model_name)

processor = MaskFormerImageProcessor(
    do_reduce_labels=False,
    size=(im_size, im_size),
    ignore_index=255,
    do_resize=False,
    do_rescale=False,
    do_normalize=False,
)

print("Модель загружена")

# Трансформации и датасет 

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

train_val_transform = A.Compose([
    A.Resize(width=512, height=512),
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.3),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])


class SegmentationDataset(Dataset):
    def __init__(self, root_folder, processor, transform=None):
        self.root_folder = root_folder
        self.processor = processor
        self.transform = transform
        self.img_dir = os.path.join(root_folder, "image") + "/"
        self.mask_dir = os.path.join(root_folder, "mask") + "/"
        self.namelist = []
        
        if os.path.exists(self.mask_dir):
            for f in os.listdir(self.mask_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    name = os.path.splitext(f)[0]
                    if (os.path.exists(self.img_dir + name + '.jpg') or
                        os.path.exists(self.img_dir + name + '.jpeg') or
                        os.path.exists(self.img_dir + name + '.png')):
                        self.namelist.append(name)
        
        self.len = len(self.namelist)
        print(f"Загружено {self.len} файлов из {root_folder}")
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        name = self.namelist[idx]
        
        # Изображение
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            path = self.img_dir + name + ext
            if os.path.exists(path):
                img_path = path
                break
        
        if img_path is None:
            raise FileNotFoundError(f"Изображение не найдено: {name}")
        
        image = Image.open(img_path).convert('RGB')
        image = np.array(image, dtype=np.uint8)
        
        # Маска
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            path = self.mask_dir + name + ext
            if os.path.exists(path):
                mask_path = path
                break
        
        if mask_path:
            mask = Image.open(mask_path)
            mask = np.array(mask, dtype=np.int64)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
        else:
            mask = np.zeros(image.shape[:2], dtype=np.int64)
        
        # Аугментации
        if self.transform:
            out = self.transform(image=image, mask=mask)
            image, mask = out['image'], out['mask']
        
        mask = (mask > 0).astype(np.int64)
        
        # Процессор
        inputs = self.processor(
            images=image,
            segmentation_maps=mask,
            return_tensors="pt"
        )
        
        # Убираем batch-размерность
        result = {}
        for k, v in inputs.items():
            if isinstance(v, list):
                result[k] = v[0] if len(v) > 0 else v
            elif isinstance(v, torch.Tensor):
                result[k] = v.squeeze(0)
            else:
                result[k] = v
        
        return result

# Даталодеры 

train_dataset = SegmentationDataset(
    '/content/test _ds_2/train',
    processor=processor,
    transform=train_val_transform
)

if len(train_dataset) == 0:
    raise ValueError("Нет данных для обучения")

indices = list(range(len(train_dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

class SubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def collate_fn(batch):
    result = {}
    for key in batch[0].keys():
        if key in ("pixel_values", "pixel_mask"):
            result[key] = torch.stack([item[key] for item in batch])
        else:
            result[key] = [item[key] for item in batch]
    return result


train_dataloader = DataLoader(
    SubsetDataset(train_dataset, train_idx),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)

val_dataloader = DataLoader(
    SubsetDataset(train_dataset, val_idx),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0
)

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

# Оптимизаторы 

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

weight_decay = 0.01
warmup_steps = 100
patience = 7

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
    betas=(0.9, 0.999)
)

warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
cosine = CosineAnnealingLR(
    optimizer,
    T_max=num_epochs * len(train_dataloader) - warmup_steps,
    eta_min=1e-6
)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

# Функции метрик 

def calculate_metrics(pred_mask, true_mask, num_classes=2):
    iou_scores, precision_scores, recall_scores = [], [], []
    
    for cls in range(num_classes):
        pred = (pred_mask == cls)
        true = (true_mask == cls)
        
        intersection = np.logical_and(pred, true).sum()
        union = np.logical_or(pred, true).sum()
        
        iou = 1.0 if union == 0 else intersection / union
        
        pred_sum = pred.sum()
        precision = 1.0 if pred_sum == 0 else intersection / pred_sum
        
        true_sum = true.sum()
        recall = 1.0 if true_sum == 0 else intersection / true_sum
        
        iou_scores.append(iou)
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    return {
        'iou': iou_scores,
        'precision': precision_scores,
        'recall': recall_scores
    }

# Цикл обучения 

for epoch in range(num_epochs):
    print(f"\nЭпоха {epoch + 1}/{num_epochs}")
    
    # Тренировка
    model.train()
    train_loss = 0.0
    train_pbar = tqdm(train_dataloader, desc="Train")
    
    for step, batch in enumerate(train_pbar):
        try:
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
            pixel_mask = batch["pixel_mask"].to(device)
            mask_labels = [m.to(device) for m in batch["mask_labels"]]
            class_labels = [c.to(device) for c in batch["class_labels"]]
            
            optimizer.zero_grad()
            
            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                mask_labels=mask_labels,
                class_labels=class_labels,
            )
            
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            avg = train_loss / (step + 1)
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{avg:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        except Exception as e:
            print(f"Ошибка в батче {step}: {e}")
            continue
    
    avg_train_loss = train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)
    print(f"Средний train loss: {avg_train_loss:.4f}")
    
    # Валидация
    model.eval()
    val_loss = 0.0
    all_metrics = []
    val_pbar = tqdm(val_dataloader, desc="Val")
    
    with torch.no_grad():
        for step, batch in enumerate(val_pbar):
            try:
                pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
                pixel_mask = batch["pixel_mask"].to(device)
                mask_labels = [m.to(device) for m in batch["mask_labels"]]
                class_labels = [c.to(device) for c in batch["class_labels"]]
                
                outputs = model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                )
                
                val_loss += outputs.loss.item()
                avg_val = val_loss / (step + 1)
                
                try:
                    pred_masks = processor.post_process_semantic_segmentation(
                        outputs,
                        target_sizes=[[512, 512]]
                    )
                    for i in range(len(pred_masks)):
                        pred = pred_masks[i].cpu().numpy()
                        true = batch["mask_labels"][i].cpu().numpy()
                        metrics = calculate_metrics(pred, true)
                        all_metrics.append(metrics)
                except Exception:
                    pass
                
                val_pbar.set_postfix({
                    'loss': f'{outputs.loss.item():.4f}',
                    'avg': f'{avg_val:.4f}'
                })
            except Exception as e:
                print(f"Ошибка валидации в батче {step}: {e}")
                continue
    
    avg_val_loss = val_loss / len(val_dataloader) if len(val_dataloader) > 0 else float('inf')
    val_losses.append(avg_val_loss)
    print(f"Средний val loss: {avg_val_loss:.4f}")
    
    if all_metrics:
        avg_iou = np.mean([m['iou'] for m in all_metrics], axis=0)
        avg_precision = np.mean([m['precision'] for m in all_metrics], axis=0)
        avg_recall = np.mean([m['recall'] for m in all_metrics], axis=0)
        
        print(f"IoU: {avg_iou[0]:.4f} (фон), {avg_iou[1]:.4f} (объект), средний: {np.mean(avg_iou):.4f}")
        print(f"Precision (объект): {avg_precision[1]:.4f}")
        print(f"Recall (объект): {avg_recall[1]:.4f}")
    
    # Сохранение лучшей модели
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
        }
        
        torch.save(checkpoint, '/content/best_model_full.pth')
        print(f"Сохранена лучшая модель (val loss {avg_val_loss:.4f})")
    else:
        patience_counter += 1
        print(f"Ухудшение. Patience: {patience_counter}/{patience}")
    
    if (epoch + 1) % 5 == 0:
        torch.save(checkpoint, f'/content/model_epoch_{epoch+1}.pth')
        print(f"Промежуточная модель сохранена (эпоха {epoch+1})")
    
    if patience_counter >= patience:
        print(f"Early stopping на эпохе {epoch+1}")
        print(f"Лучший val loss: {best_val_loss:.4f}")
        break

print(f"\nОбучение завершено. Лучший val loss: {best_val_loss:.4f}")


# График обучения 

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(train_losses, label='Train', marker='o')
axes[0].plot(val_losses, label='Validation', marker='s')
axes[0].set_xlabel('Эпоха')
axes[0].set_ylabel('Loss')
axes[0].set_title('График потерь')
axes[0].legend()
axes[0].grid(True)

if len(train_losses) == len(val_losses):
    diff = np.array(val_losses) - np.array(train_losses)
    axes[1].plot(diff, marker='x', color='red')
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('Val - Train')
    axes[1].set_title('Разница потерь')
    axes[1].grid(True)
    axes[1].axhline(y=0, color='black', linestyle='--')

plt.tight_layout()
plt.savefig('/content/training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# вызулация предсказания 

checkpoint = torch.load('/content/best_model_full.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

num_examples = min(4, len(val_dataloader.dataset))

fig, axes = plt.subplots(num_examples, 4, figsize=(20, 5 * num_examples))
if num_examples == 1:
    axes = axes.reshape(1, -1)

with torch.no_grad():
    for i in range(num_examples):
        sample = val_dataloader.dataset[i]
        pixel_values = sample["pixel_values"].unsqueeze(0).to(device, dtype=torch.float32)
        
        outputs = model(pixel_values=pixel_values)
        pred_mask = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[[512, 512]]
        )[0].cpu().numpy()
        
        true_mask = None
        if "mask_labels" in sample:
            true_mask = sample["mask_labels"]
            if isinstance(true_mask, torch.Tensor):
                true_mask = true_mask.cpu().numpy()
            if true_mask.ndim == 3:
                true_mask = true_mask[0]
        
        # Восстанавливаем изображение
        img = sample["pixel_values"].cpu().numpy()
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Изображение {i+1}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(pred_mask, cmap='gray')
        axes[i, 1].set_title("Предсказание")
        axes[i, 1].axis('off')
        
        if true_mask is not None:
            if true_mask.ndim == 3:
                true_mask = true_mask[0] if true_mask.shape[0] == 1 else true_mask[:, :, 0]
            axes[i, 2].imshow(true_mask, cmap='gray')
            axes[i, 2].set_title("Истина")
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(img)
            axes[i, 3].imshow(pred_mask, alpha=0.3, cmap='jet')
            axes[i, 3].set_title("Наложение")
            axes[i, 3].axis('off')
        else:
            axes[i, 2].axis('off')
            axes[i, 3].axis('off')

plt.tight_layout()
plt.savefig('/content/predictions.png', dpi=150, bbox_inches='tight')
plt.show()

# Сохранение модели 

import shutil

model_save_path = "SAVED_MODEL_FINAL"
os.makedirs(model_save_path, exist_ok=True)

model.save_pretrained(model_save_path)
processor.save_pretrained(model_save_path)

if os.path.exists('/content/best_model_full.pth'):
    shutil.copy('/content/best_model_full.pth', os.path.join(model_save_path, 'checkpoint.pth'))

!zip -r saved_model_final.zip SAVED_MODEL_FINAL/

checkpoint = torch.load('/content/best_model_full.pth')
print(f"Эпох обучено: {checkpoint['epoch'] + 1}")
print(f"Лучший val loss: {checkpoint['val_loss']:.4f}")
print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")

# Итогивые метрики

from sklearn.metrics import confusion_matrix

def calculate_all_metrics(pred_mask, true_mask, num_classes=2):
    pred_flat = np.clip(pred_mask.flatten(), 0, num_classes - 1)
    true_flat = np.clip(true_mask.flatten(), 0, num_classes - 1)
    
    metrics = {
        'iou_per_class': {},
        'f1_per_class': {},
        'precision_per_class': {},
        'recall_per_class': {},
        'dice_per_class': {},
    }
    
    for cls in range(num_classes):
        pred_cls = (pred_flat == cls)
        true_cls = (true_flat == cls)
        
        tp = np.logical_and(pred_cls, true_cls).sum()
        fp = np.logical_and(pred_cls, ~true_cls).sum()
        fn = np.logical_and(~pred_cls, true_cls).sum()
        union = tp + fp + fn
        
        metrics['iou_per_class'][cls] = tp / union if union > 0 else 1.0
        metrics['precision_per_class'][cls] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics['recall_per_class'][cls] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        p = metrics['precision_per_class'][cls]
        r = metrics['recall_per_class'][cls]
        metrics['f1_per_class'][cls] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        metrics['dice_per_class'][cls] = 2 * tp / (2 * tp + fp + fn + 1e-7)
    
    metrics['mean_iou'] = np.mean(list(metrics['iou_per_class'].values()))
    metrics['mean_f1'] = np.mean(list(metrics['f1_per_class'].values()))
    metrics['mean_dice'] = np.mean(list(metrics['dice_per_class'].values()))
    metrics['accuracy'] = (pred_flat == true_flat).sum() / len(pred_flat)
    metrics['confusion_matrix'] = confusion_matrix(true_flat, pred_flat, labels=range(num_classes))
    
    return metrics


checkpoint = torch.load('/content/best_model_full.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

all_preds, all_trues = [], []

with torch.no_grad():
    for batch in tqdm(val_dataloader, desc="Оценка"):
        try:
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
            outputs = model(pixel_values=pixel_values)
            
            batch_size = pixel_values.shape[0]
            for i in range(batch_size):
                try:
                    pred_mask = processor.post_process_semantic_segmentation(
                        outputs,
                        target_sizes=[[512, 512]]
                    )[i].cpu().numpy()
                    
                    true_mask = batch["mask_labels"][i].cpu().numpy()
                    
                    if pred_mask.ndim == 3:
                        pred_mask = pred_mask[0]
                    if true_mask.ndim == 3:
                        true_mask = true_mask[0]
                    
                    if pred_mask.shape != true_mask.shape:
                        from skimage.transform import resize
                        pred_mask = resize(pred_mask, true_mask.shape, order=0, preserve_range=True).astype(np.int64)
                    
                    all_preds.append(pred_mask)
                    all_trues.append(true_mask)
                except Exception:
                    continue
        except Exception as e:
            print(f"Ошибка в батче: {e}")
            continue

print(f"Собрано примеров: {len(all_preds)}")

# Усреднение метрик
if all_preds:
    all_metrics = []
    for i in range(len(all_preds)):
        try:
            m = calculate_all_metrics(all_preds[i], all_trues[i])
            all_metrics.append(m)
        except Exception:
            continue
    
    if all_metrics:
        avg = {
            'iou_per_class': {c: np.mean([m['iou_per_class'][c] for m in all_metrics]) for c in range(2)},
            'f1_per_class': {c: np.mean([m['f1_per_class'][c] for m in all_metrics]) for c in range(2)},
            'precision_per_class': {c: np.mean([m['precision_per_class'][c] for m in all_metrics]) for c in range(2)},
            'recall_per_class': {c: np.mean([m['recall_per_class'][c] for m in all_metrics]) for c in range(2)},
            'dice_per_class': {c: np.mean([m['dice_per_class'][c] for m in all_metrics]) for c in range(2)},
            'mean_iou': np.mean([m['mean_iou'] for m in all_metrics]),
            'mean_f1': np.mean([m['mean_f1'] for m in all_metrics]),
            'mean_dice': np.mean([m['mean_dice'] for m in all_metrics]),
            'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
        }
        
        names = ['Фон', 'Объект']
        print("\nМетрики на валидации:")
        for c in range(2):
            print(f"  {names[c]}: IoU={avg['iou_per_class'][c]:.4f}, F1={avg['f1_per_class'][c]:.4f}, "
                  f"Precision={avg['precision_per_class'][c]:.4f}, Recall={avg['recall_per_class'][c]:.4f}, "
                  f"Dice={avg['dice_per_class'][c]:.4f}")
        print(f"  Средний IoU: {avg['mean_iou']:.4f}")
        print(f"  Средний F1: {avg['mean_f1']:.4f}")
        print(f"  Средний Dice: {avg['mean_dice']:.4f}")
        print(f"  Accuracy: {avg['accuracy']:.4f}")
        
        with open('/content/metrics_results.txt', 'w') as f:
            f.write(f"Количество примеров: {len(all_metrics)}\n")
            f.write(f"Mean IoU: {avg['mean_iou']:.4f}\n")
            f.write(f"Mean F1: {avg['mean_f1']:.4f}\n")
            f.write(f"Mean Dice: {avg['mean_dice']:.4f}\n")
            f.write(f"Accuracy: {avg['accuracy']:.4f}\n")
        
        print("\nМетрики сохранены в metrics_results.txt")


# 
