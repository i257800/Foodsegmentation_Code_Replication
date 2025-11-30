import os
import numpy as np
import pandas as pd
import torch
torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
import albumentations as A
from PIL import Image
import argparse
from pytorch_lightning.callbacks import EarlyStopping

from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

n_classes = 104
img_size = 256

# Argument parsing setup
def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet model with custom encoder, and co-occurrence matrix usage.")
    parser.add_argument("--encoder", type=str, default="efficientnet-b5", help="Encoder name (default: efficientnet-b5)")
    parser.add_argument("--coocu", type=str, choices=["True", "False"], default="False", help="Use co-occurrence matrix in loss calculation (default: True)")
    parser.add_argument("--decay", type=int, default=600, help="Epochs to decay learning rate (default: 100)")
    parser.add_argument("--matrix", type=str, default="food103", help="occurence matrix to use (default: food103)")
    args = parser.parse_args()
    return args

# Use argparse to handle arguments
args = parse_args()

save_dir = "/content/drive/MyDrive/foodseg_weights/"
os.makedirs(save_dir, exist_ok=True)  # create folder if it doesn't exist

# Set device
if not torch.cuda.is_available():
    device = torch.device("cpu")
    print("Current device:", device)
else:
    device = torch.device("cuda")
    print(
        "devices available:",
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    )
    print("Current device:", device, "- Type:", torch.cuda.get_device_name(0))


# Create a dataframe with id of the images without extensions (.jpg)
def create_df(image_path):
    name = []
    for dirname, _, filenames in os.walk(image_path):
        for filename in filenames:
            if filename.lower().endswith('.bmp'):
                f = filename.split('.')[:-1]
                f = '.'.join(f)
                full_path = os.path.join(dirname, filename)
                if os.path.exists(full_path):
                    name.append(f)
    return pd.DataFrame({"id": name}, index=np.arange(0, len(name))).sort_values("id").reset_index(drop=True)
train_img_path = "/content/tmp/FoodSeg103-256/Images/img_dir/train/"
train_ann_path = "/content/tmp/FoodSeg103-256/Images/ann_dir/train/"
val_img_path = "/content/tmp/FoodSeg103-256/Images/img_dir/test/"
val_ann_path = "/content/tmp/FoodSeg103-256/Images/ann_dir/test/"

X_train = create_df(train_img_path)["id"].values
X_val = create_df(val_img_path)["id"].values

print("Train Size   : ", len(X_train))
print("Val Size     : ", len(X_val))

class FoodDataset(Dataset):

    def __init__(self, img_path, mask_path, X, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.img_path + self.X[idx] + ".bmp"))
        mask = np.array(Image.open(self.mask_path + self.X[idx] + ".bmp"))

        #augmented = self.resize(image=image, mask=mask)
        #image = augmented["image"]
        #mask = augmented["mask"]

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]
        norm = A.Normalize()(image=image, mask=np.expand_dims(mask, 0))
        return norm["image"].transpose(2, 0, 1), norm["mask"]

# Dataset and DataLoader
train_dataset = FoodDataset(train_img_path, train_ann_path, X_train)
valid_dataset = FoodDataset(val_img_path, val_ann_path, X_val)

batch_size = 4
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

matrix_name = "occurrence-food103.npy" if args.matrix == "food103" else "occurrence-recipe1M.npy"
adjusted_co_matrix = np.load(matrix_name)
print("Co-occurrence matrix loaded:", matrix_name)

class UNet(pl.LightningModule):
    def __init__(self, num_classes, cooccurrence_matrix, encoder_name, coocu, learning_rate=3e-4):
        super().__init__()
        self.unet = smp.create_model(
            arch="unet",
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.encoder_name = encoder_name
        self.coocu = coocu
        self.cooccurrence_weights = cooccurrence_matrix.to(device) if coocu == "True" else None
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
        # Best model tracking
        self.best_miou = 0
        self.best_ious = None
        self.best_count = 0
        if self.coocu == "True":
            self.soft_epoch = args.decay
            print('softening epochs:',self.soft_epoch)

    def forward(self, x):
        return self.unet(x)

    def compute_loss(self, logits, targets, epoch):
        targets = targets.squeeze(1).long()  # 转换为长整型并移除通道维度
        # 转换为one-hot编码以便于计算Dice Loss
        num_classes = logits.shape[1]
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2)
        targets_one_hot = targets_one_hot.float()

        # 使用共现矩阵进行加权（如果启用）
        if self.coocu == "True":
            device = logits.device
            self.cooccurrence_weights = self.cooccurrence_weights.to(device)

            # 逐步软化共现矩阵的影响
            # epsilon = 1e-6  # 防止除零错误
            softening_factor = min(epoch / self.soft_epoch, 1)  # 从 0 到 1 逐渐软化
            adjusted_cooccurrence_weights = self.cooccurrence_weights * (1 - softening_factor) + torch.ones_like(self.cooccurrence_weights) * softening_factor
            
            batch_weights = torch.einsum("bchw,ck->bchw", targets_one_hot, adjusted_cooccurrence_weights)
            # 计算加权后的Dice Loss
            weighted_dice_loss = self.dice_loss(logits, targets) * batch_weights
            return weighted_dice_loss.mean()
        else:
            # 如果不启用共现矩阵，则直接计算Dice Loss
            return self.dice_loss(logits, targets).mean()

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self.forward(images)
        epoch = self.current_epoch  # 获取当前的 epoch
        loss = self.compute_loss(logits, masks, epoch)  # 传递 epoch 到 compute_loss
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        with torch.no_grad():
            outputs = []
            # model = model.to(device)
            for image, mask in val_dl:
                # image = image.to(device)
                mask = mask.to(device)

                output = self.unet(image.to(device))#.to(device)
                tp, fp, fn, tn = smp.metrics.get_stats(
                    torch.argmax(output, 1).unsqueeze(1),
                    mask.long(),
                    mode="multiclass",
                    num_classes=n_classes,
                )
                outputs.append({"tp": tp, "fp": fp, "fn": fn, "tn": tn})

            tp = torch.cat([x["tp"] for x in outputs])
            fp = torch.cat([x["fp"] for x in outputs])
            fn = torch.cat([x["fn"] for x in outputs])
            tn = torch.cat([x["tn"] for x in outputs])
            ious = []
            for i in range(n_classes):
                ious.append(
                    round(
                        smp.metrics.iou_score(
                            tp[:, i], fp[:, i], fn[:, i], tn[:, i], reduction="macro"
                        ).item(), 4)
                )
            miou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro").item()
            self.log("val_miou", miou, prog_bar=True, logger=True)

            if miou > self.best_miou:
                self.best_miou = miou
                self.best_ious = ious
                print(f"\nepoch:{self.current_epoch} mIoU: {miou:.4f}" )
                print("IoUs:", ious)
                # torch.save(self.unet.state_dict(), args.encoder + "_"  + args.matrix + "_" + str(args.decay) + ".pth")
                save_path = os.path.join(save_dir, f"{args.encoder}_{args.matrix}_{args.decay}.pth")
                torch.save(self.unet.state_dict(), save_path)
                print(f"Best model saved to: {save_path}")
                self.best_count = 120
            self.best_count -= 1
            if self.best_count ==0:
                print('\nNo promption in 120 epochs, training ended.')
                exit(0)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


model = UNet(
    num_classes=n_classes,
    cooccurrence_matrix=torch.Tensor(adjusted_co_matrix),
    encoder_name=args.encoder,
    coocu=args.coocu
)
try:
    model.unet.load_state_dict(torch.load(args.encoder + "_"  + args.matrix + "_" + args.decay+".pth"))
    print("Best model loaded.")
    model.to(device)
except:
    print("No best model found, training from scratch.")
    
epochs = 500
print('epochs:', epochs)

early_stop_callback = EarlyStopping(
    monitor="val_miou",   # metric to monitor
    mode="max",           # maximize mIoU
    patience=50,          # stop if no improvement in 50 epochs
    verbose=True
)

resume_path = os.path.join(save_dir, "efficientnet-b0_food103_6000.pth")

if os.path.exists(resume_path):
    print("\nResuming training from:", resume_path)
    ckpt = torch.load(resume_path, map_location=device)
    model.unet.load_state_dict(ckpt)
    model.current_epoch = 17   # start from epoch 17
else:
    print("No pretrained checkpoint found. Training from scratch.")


trainer = pl.Trainer(accelerator="gpu",
                    max_epochs=epochs,
                    precision='bf16-mixed',
                    num_sanity_val_steps=0,
                    logger=False,
                    enable_checkpointing=False,
                    enable_progress_bar=False,
                    callbacks=[early_stop_callback]   # add EarlyStopping
                    )

print('\n')
print(args.encoder, args.coocu)
trainer.fit(model, train_dl, val_dl)
