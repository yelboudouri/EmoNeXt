import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import wandb
from models import get_model
from scheduler import CosineAnnealingWithWarmRestartsLR

seed = 2001
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self,
                 model,
                 training_dataloader,
                 validation_dataloader,
                 testing_dataloader,
                 classes,
                 output_dir,
                 max_epochs: int = 10000,
                 early_stopping_patience: int = 12,
                 execution_name=None,
                 lr: float = 1e-4,
                 amp: bool = False,
                 ema_decay: float = 0.99,
                 ema_update_every: int = 16,
                 gradient_accumulation_steps: int = 1,
                 checkpoint_path: str = None,
                 ):

        self.epochs = max_epochs

        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.testing_dataloader = testing_dataloader

        self.classes = classes
        self.num_classes = len(classes)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device used: " + self.device.type)

        self.amp = amp
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.model = model.to(self.device)

        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.scheduler = CosineAnnealingWithWarmRestartsLR(self.optimizer, warmup_steps=128, cycle_steps=1024)
        self.ema = EMA(model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        self.early_stopping_patience = early_stopping_patience

        self.output_directory = Path(output_dir)
        self.output_directory.mkdir(exist_ok=True)

        self.best_val_accuracy = 0

        self.execution_name = 'model' if execution_name is None else execution_name

        if checkpoint_path:
            self.load(checkpoint_path)

        wandb.watch(model, log='all')

    def run(self):
        counter = 0  # Counter for epochs with no validation loss improvement

        images, _ = next(iter(self.training_dataloader))
        images = [transforms.ToPILImage()(image) for image in images]
        wandb.log({
            'Images': [wandb.Image(image) for image in images]
        })

        for epoch in range(self.epochs):
            print("[Epoch: %d/%d]" % (epoch + 1, self.epochs))

            self.visualize_stn()
            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.val_epoch()

            wandb.log({'Train Loss': train_loss,
                       'Val Loss': val_loss,
                       'Train Accuracy': train_accuracy,
                       'Val Accuracy': val_accuracy,
                       'Epoch': epoch + 1})

            # Early stopping
            if val_accuracy > self.best_val_accuracy:
                self.save()
                counter = 0
                self.best_val_accuracy = val_accuracy
            else:
                counter += 1
                if counter >= self.early_stopping_patience:
                    print(
                        "Validation loss did not improve for %d epochs. Stopping training." % self.early_stopping_patience)
                    break

        self.test_model()
        wandb.finish()

    def train_epoch(self):
        self.model.train()

        avg_accuracy = []
        avg_loss = []

        pbar = tqdm(unit="batch", file=sys.stdout, total=len(self.training_dataloader))
        for batch_idx, data in enumerate(self.training_dataloader):
            inputs, labels = data

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.autocast(self.device.type, enabled=self.amp):
                predictions, _, loss = self.model(inputs, labels)

            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                self.ema.update()
                self.scheduler.step()

            batch_accuracy = (predictions == labels).sum().item() / labels.size(0)

            avg_loss.append(loss.item())
            avg_accuracy.append(batch_accuracy)

            # Update progress bar
            pbar.set_postfix({'loss': np.mean(avg_loss), 'acc': np.mean(avg_accuracy) * 100.0})
            pbar.update(1)

        pbar.close()

        return np.mean(avg_loss), np.mean(avg_accuracy) * 100.0

    def val_epoch(self):
        self.model.eval()

        avg_loss = []
        predicted_labels = []
        true_labels = []

        pbar = tqdm(unit="batch", file=sys.stdout, total=len(self.validation_dataloader))
        for batch_idx, (inputs, labels) in enumerate(self.validation_dataloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.autocast(self.device.type, enabled=self.amp):
                predictions, _, loss = self.model(inputs, labels)

            avg_loss.append(loss.item())
            predicted_labels.extend(predictions.tolist())
            true_labels.extend(labels.tolist())

            pbar.update(1)

        pbar.close()

        accuracy = torch.eq(torch.tensor(predicted_labels), torch.tensor(true_labels)).float().mean().item()
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                                                                   y_true=true_labels,
                                                                   preds=predicted_labels,
                                                                   class_names=self.classes)})

        print('Eval loss: %.4f, Eval Accuracy: %.4f %%' % (np.mean(avg_loss) * 1.0, accuracy * 100.0))
        return np.mean(avg_loss), accuracy * 100.0

    def test_model(self):
        self.ema.eval()

        predicted_labels = []
        true_labels = []

        pbar = tqdm(unit="batch", file=sys.stdout, total=len(self.testing_dataloader))
        for batch_idx, (inputs, labels) in enumerate(self.testing_dataloader):
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w)

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.autocast(self.device.type, enabled=self.amp):
                _, logits = self.ema(inputs)
            outputs_avg = logits.view(bs, ncrops, -1).mean(1)
            predictions = torch.argmax(outputs_avg, dim=1)

            predicted_labels.extend(predictions.tolist())
            true_labels.extend(labels.tolist())

            pbar.update(1)

        pbar.close()

        accuracy = torch.eq(torch.tensor(predicted_labels), torch.tensor(true_labels)).float().mean().item()
        print('Test Accuracy: %.4f %%' % (accuracy * 100.0))

        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                                                                   y_true=true_labels,
                                                                   preds=predicted_labels,
                                                                   class_names=self.classes)})

    def visualize_stn(self):
        self.model.eval()

        batch = torch.utils.data.Subset(val_dataset, range(32))

        # Access the batch data
        batch = torch.stack([batch[i][0] for i in range(len(batch))]).to(self.device)
        with torch.autocast(self.device.type, enabled=self.amp):
            stn_batch = self.model.stn(batch)

        to_pil = transforms.ToPILImage()

        grid = to_pil(torchvision.utils.make_grid(batch, nrow=16, padding=4))
        stn_batch = to_pil(torchvision.utils.make_grid(stn_batch, nrow=16, padding=4))

        wandb.log({'batch': wandb.Image(grid), 'stn': wandb.Image(stn_batch)})

    def save(self):
        data = {
            'model': self.model.state_dict(),
            'opt': self.optimizer.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.scaler.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_acc': self.best_val_accuracy,
        }

        torch.save(data, str(self.output_directory / f'{self.execution_name}.pt'))

    def load(self, path):
        data = torch.load(path, map_location=self.device)

        self.model.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.scaler.load_state_dict(data['scaler'])
        self.scheduler.load_state_dict(data['scheduler'])
        self.best_val_accuracy = data['best_acc']


def plot_images():
    # Create a grid of images for visualization
    num_rows = 4
    num_cols = 8
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5))

    # Plot the images
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j  # Calculate the corresponding index in the dataset
            image, _ = train_dataset[index]  # Get the image
            axes[i, j].imshow(image.permute(1, 2, 0))  # Convert tensor to PIL image format and plot
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig("images.png")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train EmoNeXt on Fer2013")

    parser.add_argument("--dataset-path", type=str, help="Path to the dataset")
    parser.add_argument("--output-dir", type=str, default="out",
                        help="Path where the best model will be saved")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--amp', action='store_true', default=False, help='Enable mixed precision training')
    parser.add_argument('--in_22k', action='store_true', default=False)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Number of steps to accumulate gradients before updating the model weights')
    parser.add_argument("--num-workers", type=int, default=0,
                        help="The number of subprocesses to use for data loading."
                             "0 means that the data will be loaded in the main process.")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to the checkpoint file for resuming training or performing inference')
    parser.add_argument('--model-size', choices=['tiny', 'small', 'base', 'large', 'xlarge'], default='tiny',
                        help='Choose the size of the model: tiny, small, base, large, or xlarge')

    opt = parser.parse_args()
    print(opt)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exec_name = f"EmoNeXt_{opt.model_size}_{current_time}"

    wandb.init(project="EmoNeXt", name=exec_name, anonymous="must")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Grayscale(),
        transforms.Resize(236),
        transforms.RandomRotation(degrees=20),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(236),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(236),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([crop.repeat(3, 1, 1) for crop in crops])),
    ])

    train_dataset = datasets.ImageFolder(opt.dataset_path + '/train', train_transform)
    val_dataset = datasets.ImageFolder(opt.dataset_path + '/val', val_transform)
    test_dataset = datasets.ImageFolder(opt.dataset_path + '/test', test_transform)

    print("Using %d images for training." % len(train_dataset))
    print("Using %d images for evaluation." % len(val_dataset))
    print("Using %d images for testing." % len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    net = get_model(
        len(train_dataset.classes),
        opt.model_size,
        in_22k=opt.in_22k
    )

    Trainer(
        model=net,
        training_dataloader=train_loader,
        validation_dataloader=val_loader,
        testing_dataloader=test_loader,
        classes=train_dataset.classes,
        execution_name=exec_name,
        lr=opt.lr,
        output_dir=opt.output_dir,
        checkpoint_path=opt.checkpoint,
        max_epochs=opt.epochs,
        amp=opt.amp
    ).run()
