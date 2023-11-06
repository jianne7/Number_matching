import os
import random
import argparse
import time
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score

from transformers import AdamW, get_scheduler, AutoTokenizer, AutoModel, ElectraModel

from utils import NumberClassification
from NumClassDataModule import NumClassDataset

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_logger(name: str, file_path: str, stream=False) -> logging.RootLogger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path)

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if stream:
        logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def make_directory(directory: str) -> str:
    """
    경로가 없으면 생성
    Args:
        directory (str): 새로 만들 경로
    Returns:
        str: 상태 메시지
    """
    try:
        if not os.path.isdir(directory):
            os.makedirs(directory)
            msg = f"Create directory {directory}"

        else:
            msg = f"{directory} already exists"

    except OSError as e:
        msg = f"Fail to create directory {directory} {e}"

    return msg


def save_checkpoint(epoch, model, optimizer, train_step, valid_step, savepath):
    state = {'epoch' : epoch,
             'state_dict' : model.state_dict(),
             'optimizer' : optimizer.state_dict(),
             'train_step' : train_step,
             'valid_step' : valid_step
            }
    torch.save(state, savepath)


def get_label_class(tokenizer, labels, candidate_num_list):
    label_class_list = []
    for label, candidate_num in zip(labels, candidate_num_list):
        # remove -100 padding
        pad_ind = torch.where(label == -100)[0][0]
        label_id = label[:pad_ind][1:-1] # for except [CLS], and last [SEP]
        # find lable class in candidate numbers
        try:
            label_class = candidate_num.index(tokenizer.decode(label_id)) # 후보 숫자표현들 중 정답 클래스 인덱스
        except:
            label_class = candidate_num.index('[SEP]') # 만약 라벨이 숫자 후보에 없는 경우 답없음으로

        label_class_list.append(label_class)
    
    return label_class_list


def train(model, number_classifier, device, tokenizer, logger, dataloader, optimizer, criterion, train_global_steps, writer):
    logger.info("Start Training!!!!")
    model.train()

    train_loss = []
    train_acc = []
    total_num = 0
    train_step = 0
    for batch in dataloader:
        start = time.time()
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'])
        
        last_hidden = outputs.last_hidden_state
        output, candidate_num_list, _, _ = number_classifier(last_hidden, batch['input_ids'], batch['numbers'])
        
        target = get_label_class(tokenizer, batch['labels'], candidate_num_list)
        target = torch.LongTensor(target).to(device)
        loss = criterion(output, target)
        
        preds = [torch.argmax(i).item() for i in output]
        acc = accuracy_score(target.detach().cpu(), preds)

        loss.backward()
        optimizer.step()
        # lr_scheduler.step()        
        progress_bar.update(1)

        train_loss.append(loss.item())
        train_acc.append(acc)
        total_num += 1

        logger.info(f'loss : {loss.item():.4f} | accuracy : {acc:.4f} | second : {(time.time() - start):.2f}s')
        if (total_num+train_global_steps) % 100 == 99:
            writer.add_scalar('Train/loss', np.mean(train_loss), (total_num+train_global_steps+1))
            writer.add_scalar('Train/accuracy', np.mean(train_acc), (total_num+train_global_steps+1))
            train_step = total_num+train_global_steps+1

    train_global_steps+=total_num
    avg_loss = np.mean(train_loss)
    avg_acc = np.mean(train_acc)

    return avg_loss, avg_acc, train_step


def valid(model, number_classifier, device, tokenizer, logger, dataloader, criterion, valid_global_steps, writer):
    logger.info("Start Validation!!!!")  
    model.eval()

    valid_loss = []
    valid_acc = []
    total_num = 0
    valid_step = 0
    with torch.no_grad():
        for batch in dataloader:
            start = time.time()

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'])
            
            last_hidden = outputs.last_hidden_state
            output, candidate_num_list, _, _ = number_classifier(last_hidden, batch['input_ids'], batch['numbers'])

            target = get_label_class(tokenizer, batch['labels'], candidate_num_list)
            target = torch.LongTensor(target).to(device)
            loss = criterion(output, target)
            
            preds = [torch.argmax(i).item() for i in output]
            acc = accuracy_score(target.detach().cpu(), preds)

            valid_loss.append(loss.item())
            valid_acc.append(acc)
            total_num += 1

            logger.info(f'loss : {loss.item():.4f} | accuracy : {acc:.4f} | second : {(time.time() - start):.2f}s')
            if (total_num+valid_global_steps) % 100 == 99:
                writer.add_scalar('Valid/loss', np.mean(valid_loss), (total_num+valid_global_steps+1))
                writer.add_scalar('Valid/accuracy', np.mean(valid_acc), (total_num+valid_global_steps+1))
                valid_step = total_num+valid_global_steps+1

    valid_global_steps+=total_num
    avg_loss = np.mean(valid_loss)
    avg_acc = np.mean(valid_acc)

    return avg_loss, avg_acc, valid_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=586)
    parser.add_argument('--model_name', type=str, default='roberta') # roberta, kb-albert, koelectra
    # parser.add_argument('--model', type=str, default='kb-albert-char-base-v2') # klue/roberta-base
    # parser.add_argument('--tokenizer', type=str, default='kb-albert-char-base-v2') # klue/roberta-base
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16) # 16
    parser.add_argument('--lr', type=float, default=3e-5, help='the learning rate')   
    parser.add_argument('--epochs', type=int, default=30) # 30
    parser.add_argument('--num_workers', type=int, default=8)     
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--set_device', type=int, default=0, help='setting the gpu number')
    parser.add_argument('--data_path', type=str, default='./data/v3')
    parser.add_argument('--ckpt_path', type=str, default=None, help='load ckpt to continue trainig')
    parser.add_argument('--tb_logs', type=str, default=None, help='tensorboard log file path to continue training')
    parser.add_argument('--version', type=int, default=3, help='train version')
    parser.add_argument('--mask', type=bool, default=False, help='number token mask')
    args = parser.parse_args()

    set_random_seed(random_seed=args.seed)

    device = torch.device(args.device)
    # torch.cuda.set_device(args.set_device)
    print(f'Current cuda device: {torch.cuda.current_device()}')

    make_directory('log')
    logger = get_logger(name='train',
                file_path=os.path.join('log', f'train_v{args.version}_{args.model_name}.log'), stream=True)
    logger.info(f'Arguments : {args}')

    if args.tb_logs is None:
        writer = SummaryWriter(f'runs/train_v{args.version}_{args.model_name}')
    else:
        writer = SummaryWriter(args.tb_logs) # 이어서 학습할 때 SummaryWriter(괄호 안에 이전 파일 주소 넣으면 됨)

    if args.model_name == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
        model = AutoModel.from_pretrained("klue/roberta-base")
    elif args.model_name == 'kb-albert':
        tokenizer = AutoTokenizer.from_pretrained("kb-albert-char-base-v2")
        model = AutoModel.from_pretrained("kb-albert-char-base-v2")
    elif args.model_name == 'koelectra':
        tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")

    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # model = AutoModel.from_pretrained(args.model)
    model.to(device)

    number_classifier = NumberClassification(device, tokenizer, args)

    train_dataset = NumClassDataset(dirpath=args.data_path,
                                    mode='train',
                                    tokenizer=tokenizer,
                                    args=args)
    
    train_dataloader = DataLoader(train_dataset,
                                  num_workers=args.num_workers,
                                  batch_size=args.batch_size,
                                  shuffle=True)

    valid_dataset = NumClassDataset(dirpath=args.data_path,
                                    mode='valid',
                                    tokenizer=tokenizer,
                                    args=args) 
    
    valid_dataloader = DataLoader(valid_dataset,
                                  num_workers=args.num_workers,
                                  batch_size=args.batch_size,
                                  shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    num_training_steps = args.epochs * len(train_dataloader)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150)

    print(num_training_steps)
    progress_bar = tqdm(range(num_training_steps))

    # load checkpoints
    if args.ckpt_path is not None:
        checkpoint = torch.load(args.ckpt_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        restart_epoch = checkpoint['epoch']+1
        train_overleap = checkpoint['train_step']
        valid_overleap = checkpoint['valid_step']
        print('Continue to training...')
    else:
        restart_epoch, train_overleap, valid_overleap = 0, 0, 0    

    train_global_steps = 0 + train_overleap
    valid_global_steps = 0 + valid_overleap
    eval_loss = 1e8
    for epoch in range(restart_epoch, args.epochs):
        epoch_start = time.time()

        train_avg_loss, train_avg_acc, train_step = train(model, number_classifier, device, tokenizer, logger, train_dataloader, optimizer, criterion, train_global_steps, writer)
        writer.add_scalar('Epoch_Train/loss', train_avg_loss, epoch)
        writer.add_scalar('Epoch_Train/accuracy', train_avg_acc, epoch)

        logger.info('='*100)
        logger.info(f'Train | epoch: {epoch:02d} | loss: {train_avg_loss:.4f} | accuracy : {train_avg_acc:.4f}')
        logger.info('='*100)

        valid_avg_loss, valid_avg_acc, valid_step= valid(model, number_classifier, device, tokenizer, logger, valid_dataloader, criterion, valid_global_steps, writer)
        writer.add_scalar('Epoch_Valid/loss', valid_avg_loss, epoch)
        writer.add_scalar('Epoch_Valid/accuracy', valid_avg_acc, epoch)

        dir_path = f'checkpoints/{args.model_name}/train_v{args.version}'
        make_directory(dir_path)
        savepath = f'./{dir_path}/epoch={epoch+1}_loss={valid_avg_loss:.4f}_acc={valid_avg_acc:.4f}.ckpt'
        save_checkpoint(epoch, model, optimizer, train_step, valid_step, savepath)
        logger.info(f'=======================Saved Checkpoints!!!!=======================')

        if eval_loss >= valid_avg_loss:
            # dir_path = f'checkpoints/train_v{args.version}'
            # make_directory(dir_path)
            savepath = f'./{dir_path}/best.ckpt' # save the last checkpoint
            # savepath = f'./{dir_path}/epoch={epoch}_loss={valid_avg_loss:.4f}_acc={valid_avg_acc:.4f}.ckpt' # save the best checkpoint
            save_checkpoint(epoch, model, optimizer, train_step, valid_step, savepath)
            eval_loss = valid_avg_loss # update by average loss
            logger.info(f'=======================Saved BEST Checkpoints!!!! (epoch={epoch+1})=======================')
        
        logger.info('='*100)
        logger.info(f'Valid | epoch: {epoch:02d} | loss: {valid_avg_loss:.4f} | accuracy : {valid_avg_acc:.4f} | one epoch second : {(time.time() - epoch_start):.2f}s')
        logger.info('='*100)
        
        lr_scheduler.step()


    print('Finish!!!!')
   