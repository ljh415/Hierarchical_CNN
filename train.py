import os
import wandb
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from dataset import Cifar100Dataset
from model import HierarchyCNN
from utils import get_now, checkpoint_save

def main(args):
    
    # wandb init
    if args.wandb:
        if args.proj_name:
            wandb_name = f"{args.run_name}_{get_now(time=True)}"
        else :
            wandb_name = f"HierarachyCNN_{get_now(time=True)}"
        
        wandb.init(entity="ljh415", project=args.proj_name, dir='/home/jaeho/hdd/wandb', name=wandb_name)
    
    # checkpoint save directory
    save_dir = os.path.join(args.ckpt_savedir, args.proj_name)
    if args.run_name:
        save_dir = os.path.join(save_dir, f"{args.run_name}_{get_now(time=True)}")
    else :
        save_dir = os.path.join(save_dir, f"HierarachyCNN_{get_now(time=True)}")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    ####
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # acc
    accuracy = Accuracy().to(device)
    
    # dataloader
    train_dataset = Cifar100Dataset(args.data_path, 'train', transform)
    div_dataset = Cifar100Dataset(args.data_path, 'test', transform)
    valid_dataset, test_dataset = torch.utils.data.random_split(div_dataset, [5000, 5000])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    coarse_criterion = nn.CrossEntropyLoss().to(device)
    fine_criterion = nn.CrossEntropyLoss().to(device)
    
    model = HierarchyCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    print("Start Training")
    
    for epoch in range(args.epochs):
        model.train()
        
        step = 0
        running_train_loss = 0
        running_train_coarse_loss, running_train_fine_loss = 0, 0
        running_train_coarse_acc, running_train_fine_acc = 0, 0
        now_lr = lr_scheduler.optimizer.param_groups[0]['lr']
        
        for batch_idx, (img_batch, coarse_batch, fine_batch) in enumerate(train_dataloader):
            optimizer.zero_grad()
            img_batch = img_batch.to(device)
            coarse_batch = coarse_batch.to(device)
            fine_batch = fine_batch.to(device)
            
            coarse_out, fine_out = model(img_batch)
            
            # acc
            batch_coarse_acc = accuracy(coarse_out, coarse_batch)
            batch_fine_acc = accuracy(fine_out, fine_batch)
            running_train_coarse_acc += batch_coarse_acc.detach().cpu().item()
            running_train_fine_acc += batch_fine_acc.detach().cpu().item()
            
            # loss
            coarse_loss = coarse_criterion(coarse_out, coarse_batch)
            fine_loss = fine_criterion(fine_out, fine_batch)
            
            loss = coarse_loss + fine_loss*2
            
            # loss calc
            running_train_loss += loss.detach().cpu()
            running_train_coarse_loss += coarse_loss.detach().cpu()
            running_train_fine_loss += fine_loss.detach().cpu()
            
            loss.backward()
            optimizer.step()
            step+=1
            
            status = (
                "\r> epoch: {:3d} > step: {:3d} > loss: {:.3f}, coarse loss: {:.3f}, fine loss: {:.3f}, coarse acc: {:.3f}, fine acc: {:.3f}, lr: {}  ".format(
                                epoch+1,
                                step,
                                running_train_loss/(batch_idx+1),
                                running_train_coarse_loss/(batch_idx+1),
                                running_train_fine_loss/(batch_idx+1),
                                running_train_coarse_acc/(batch_idx+1),
                                running_train_fine_acc/(batch_idx+1),
                                now_lr,
                            )
            )
            
            print(status, end="")
        print()
        
        train_loss = running_train_loss / len(train_dataloader)
        train_coarse_loss = running_train_coarse_loss / len(train_dataloader)
        train_fine_loss = running_train_fine_loss / len(train_dataloader)
        
        train_coarse_acc = running_train_coarse_acc / len(train_dataloader)
        train_fine_acc = running_train_fine_acc / len(train_dataloader)
        
        # valid
        running_val_loss = 0
        running_val_coarse_loss, running_val_fine_loss = 0, 0
        running_val_coarse_acc, running_val_fine_acc = 0, 0
        
        with torch.no_grad():
            model.eval()
            for img_batch, coarse_batch, fine_batch in valid_dataloader:
                img_batch = img_batch.to(device)
                coarse_batch = coarse_batch.to(device)
                fine_batch = fine_batch.to(device)

                coarse_out, fine_out = model(img_batch)
            
                # acc
                batch_coarse_acc = accuracy(coarse_out, coarse_batch)
                batch_fine_acc = accuracy(fine_out, fine_batch)
                
                running_val_coarse_acc += batch_coarse_acc.detach().cpu().item()
                running_val_fine_acc += batch_fine_acc.detach().cpu().item()
                
                # loss
                coarse_val_loss = coarse_criterion(coarse_out, coarse_batch)
                fine_val_loss = fine_criterion(fine_out, fine_batch)
                
                val_loss = coarse_val_loss + fine_val_loss*2
                
                # loss calc
                running_val_loss += val_loss.detach().cpu()
                running_val_coarse_loss += coarse_val_loss.detach().cpu()
                running_val_fine_loss += fine_val_loss.detach().cpu()
        
        validation_loss = running_val_loss / len(valid_dataloader)
        validation_coarse_loss = running_val_coarse_loss / len(valid_dataloader)
        validation_fine_loss = running_val_fine_loss / len(valid_dataloader)
        
        validation_coarse_acc = running_val_coarse_acc / len(valid_dataloader)
        validation_fine_acc = running_val_fine_acc / len(valid_dataloader)
        
        status = (
            "> Validation loss: {:.3f}, coarse loss: {:.3f}, fine loss: {:.3f}, coarse acc: {:.3f}, fine acc: {:.3f}\n ".format(
                            validation_loss,
                            validation_coarse_loss,
                            validation_fine_loss,
                            validation_coarse_acc,
                            validation_fine_acc
                        )
        )
        
        print(status, end="")
        
        if epoch % args.freq_checkpoint == 0:
            checkpoint_save(model, save_dir, epoch, validation_loss)
        
        if args.wandb:
            wandb.log({
                "lr" : now_lr,
                "train_loss": train_loss,
                "train_coarse_loss": train_coarse_loss,
                "train_fine_loss": train_fine_loss,
                "train_coarse_acc": train_coarse_acc,
                "train_fine_acc": train_fine_acc,
                "valid_loss": validation_loss,
                "valid_coarse_loss": validation_coarse_loss,
                "valid_fine_loss": validation_fine_loss,
                "valid_coarse_acc": validation_coarse_acc,
                "valid_fine_acc": validation_fine_acc
            })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/jaeho/hdd/datasets/cifar-100-python")
    parser.add_argument("--ckpt_savedir", type=str, default="/home/jaeho/hdd/ckpt")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--proj_name", type=str, default="Classification_2022")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--freq_checkpoint", type=int, default=1)
    
    args = parser.parse_args()
    
    main(args)