import clip
import numpy as np
import os
import shutil
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from types import SimpleNamespace


TEMPLATE = "This is a photo of a {}"
ROOT = "./data"
NUM_WORKERS = 2
BATCH_SIZE = 256

EPOCHS = 5
LEARNING_RATE = 40
MOMENTUM = 0.9
WEIGHT_DECAY = 0
WARMUP = 100
PATIENCE = 2

MODEL_DIR = "./save/models"


def save_checkpoint(state, is_best=False, filename='checkpoint.pth.tar'):
    savefile = os.path.join(MODEL_DIR, filename)
    bestfile = os.path.join(MODEL_DIR, 'model_best.pth.tar')
    torch.save(state, savefile)
    if is_best:
        shutil.copyfile(savefile, bestfile)
        print('saved best file')


def accuracy(output, target, topk=(1,)):
    """Computes accuracy over k top predictions for specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


class PadPrompter(nn.Module):
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size
        self.device = args.device

        self.base_size = image_size - pad_size*2
        self.pad_up = nn.Parameter(torch.randn(
            [1, 3, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn(
            [1, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn(
            [1, 3, image_size - pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn(
            [1, 3, image_size - pad_size*2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.base_size).to(self.device)
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])

        return x + prompt


def train(train_loader, texts, model, prompter, optimizer, scheduler,
          criterion, scaler, epoch, device):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    prompter.train()
    num_batches_per_epoch = len(train_loader)
    for i, (images, target) in enumerate(tqdm(train_loader)):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        optimizer.zero_grad()
        images = images.to(device)
        target = target.to(device)
        text_tokens = clip.tokenize(texts).to(device)

        with autocast():
            prompted_images = prompter(images)
            output, _ = model(prompted_images, text_tokens)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

    print(f"Epoch: {epoch}, Loss: {losses.avg}, Top1: {top1.avg}")
    return losses.avg, top1.avg


def validate(val_loader, texts, model, prompter, criterion, device):
    losses = AverageMeter('Loss', ':.4e')
    top1_org = AverageMeter('Original Acc@1', ':6.2f')
    top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')
    prompter.eval()

    with torch.no_grad():
        for _, (images, target) in enumerate(tqdm(val_loader)):

            images = images.to(device)
            target = target.to(device)
            text_tokens = clip.tokenize(texts).to(device)
            prompted_images = prompter(images)

            # compute output
            output_prompt, _ = model(prompted_images, text_tokens)
            output_org, _ = model(images, text_tokens)
            loss = criterion(output_prompt, target)

            # measure accuracy and record loss
            acc1 = accuracy(output_prompt, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1_prompt.update(acc1[0].item(), images.size(0))

            acc1 = accuracy(output_org, target, topk=(1,))
            top1_org.update(acc1[0].item(), images.size(0))

    print(f"validation accuracy is {top1_prompt.avg}")
    return top1_prompt.avg


def main():
    # TODO: set seed
    best_acc1 = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    model.eval()
    args = SimpleNamespace(prompt_size=30, image_size=224, device=device)
    prompter = PadPrompter(args).to(device)
    train_dataset = CIFAR10(ROOT, transform=preprocess, download=True,
                            train=True)
    val_dataset = CIFAR10(ROOT, transform=preprocess, download=True,
                            train=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              pin_memory=True, num_workers=NUM_WORKERS,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              pin_memory=True, num_workers=NUM_WORKERS,
                              shuffle=True)
    class_names = train_dataset.classes
    texts = [TEMPLATE.format(label) for label in class_names]
    print(texts)
    optimizer = torch.optim.SGD(prompter.parameters(),
                                lr = LEARNING_RATE,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    scaler = GradScaler()
    total_steps = len(train_loader) * EPOCHS
    scheduler = cosine_lr(optimizer, LEARNING_RATE, WARMUP, total_steps)
    print(total_steps)
    cudnn.benchmark = True
    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    epochs_since_improvement = 0
    for epoch in range(EPOCHS):
        train(train_loader, texts, model, prompter, optimizer, scheduler,
              criterion, scaler, epoch, device)
        acc1 = validate(val_loader, texts, model, prompter, criterion, device)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': prompter.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best=is_best)
        if is_best:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print(f"No improvement for {epochs_since_improvement} epochs.")

        if epochs_since_improvement > PATIENCE:
            print("The training halted by early stopping criterion.")
            break


if __name__ == "__main__":
    main()
