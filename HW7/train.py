import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from data import get_dataloader
from model import StudentNet

def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    # 一般的Cross Entropy
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    # 讓logits的log_softmax對目標機率(teacher的logits/T後softmax)做KL Divergence。
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss

# get dataloader
train_dataloader = get_dataloader('training', batch_size=32)
valid_dataloader = get_dataloader('validation', batch_size=32)

#!gdown --id '1B8ljdrxYXJsZv2vmTequdPOofp3VF3NN' --output teacher_resnet18.bin

teacher_net = models.resnet18(pretrained=False, num_classes=11).cuda()
student_net = StudentNet(base=16).cuda()

teacher_net.load_state_dict(torch.load(f'./teacher_resnet18.bin'))
optimizer = optim.AdamW(student_net.parameters(), lr=5e-4)

def run_epoch(dataloader, update=True, alpha=0.5):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        # 清空 optimizer
        optimizer.zero_grad()
        # 處理 input
        inputs, hard_labels = batch_data
        inputs = inputs.cuda()
        hard_labels = torch.LongTensor(hard_labels).cuda()
        # 因為Teacher沒有要backprop，所以我們使用torch.no_grad
        # 告訴torch不要暫存中間值(去做backprop)以浪費記憶體空間。
        with torch.no_grad():
            soft_labels = teacher_net(inputs)

        if update:
            logits = student_net(inputs)
            # 使用我們之前所寫的融合soft label&hard label的loss。
            # T=20是原始論文的參數設定。
            loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            loss.backward()
            optimizer.step()    
        else:
            # 只是算validation acc的話，就開no_grad節省空間。
            with torch.no_grad():
                logits = student_net(inputs)
                loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            
        total_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
        total_num += len(inputs)

        total_loss += loss.item() * len(inputs)
    return total_loss / total_num, total_hit / total_num

student_net.load_state_dict(torch.load('student_model.bin'))
# TeacherNet永遠都是Eval mode.
teacher_net.eval()
now_best_acc = 0
for epoch in range(300):
    student_net.train()
    train_loss, train_acc = run_epoch(train_dataloader, update=True, alpha = 0.2)
    student_net.eval()
    valid_loss, valid_acc = run_epoch(valid_dataloader, update=False, alpha = 0.2)

    # 存下最好的model。
    if valid_acc > now_best_acc:
        now_best_acc = valid_acc
        torch.save(student_net.state_dict(), 'student_model.bin')
    print('epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} valid loss: {:6.4f}, acc {:6.4f}'.format(
        epoch, train_loss, train_acc, valid_loss, valid_acc))
