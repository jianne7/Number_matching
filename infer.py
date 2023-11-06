import os
import re
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModel, ElectraModel

from utils import NumberClassification
from NumClassDataModule import NumClassDataset
# from kobert_tokenizer import KoBERTTokenizer

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

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


# def get_label_class(tokenizer, labels, candidate_num_list):
#     label_class_list = []
#     decoded_label_list = []
#     for label, candidate_num in zip(labels, candidate_num_list):
#         # remove -100 padding
#         pad_ind = torch.where(label == -100)[0][0]
#         label_id = label[:pad_ind][1:-1] # for except [CLS], and last [SEP]
#         # find lable class in candidate numbers
#         decoded_label = tokenizer.decode(label_id)
#         label_class = candidate_num.index(decoded_label) # 후보 숫자표현들 중 정답 클래스 인덱스
#         label_class_list.append(label_class)
#         decoded_label_list.append(decoded_label)
    
#     return label_class_list, decoded_label_list

def get_label_class(tokenizer, labels, candidate_num_list):
    label_class_list = []
    decoded_label_list = []
    for label, candidate_num in zip(labels, candidate_num_list):
        # remove -100 padding
        pad_ind = torch.where(label == -100)[0][0]
        label_id = label[:pad_ind][1:-1] # for except [CLS], and last [SEP]
        decoded_label = tokenizer.decode(label_id)
        # find lable class in candidate numbers
        try:
            label_class = candidate_num.index(tokenizer.decode(label_id)) # 후보 숫자표현들 중 정답 클래스 인덱스
        except:
            label_class = candidate_num.index('[SEP]') # 만약 라벨이 숫자 후보에 없는 경우 답없음으로

        label_class_list.append(label_class)
        decoded_label_list.append(decoded_label)
    
    return label_class_list, decoded_label_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=586)
    parser.add_argument('--model_name', type=str, default='roberta') # klue/roberta-base
    # parser.add_argument('--model', type=str, default='klue/roberta-base')
    # parser.add_argument('--tokenizer', type=str, default='klue/roberta-base')
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16) # 16
    parser.add_argument('--num_workers', type=int, default=8)     
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--set_device', type=int, default=1, help='setting the gpu number')
    parser.add_argument('--data_path', type=str, default='./data/v3')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints', help='load checkpoint')
    parser.add_argument('--version', type=int, default=3, help='train version')
    parser.add_argument('--mask', type=bool, default=False, help='number token mask')
    args = parser.parse_args()
    set_random_seed(random_seed=args.seed)

    device = torch.device(args.device)
    # torch.cuda.set_device(args.set_device)
    print(f'Current cuda device: {torch.cuda.current_device()}')
    print(f'Arguments: {args}')
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # model = AutoModel.from_pretrained(args.model)

    if args.model_name == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
        model = AutoModel.from_pretrained("klue/roberta-base")
    elif args.model_name == 'kb-albert':
        tokenizer = AutoTokenizer.from_pretrained("kb-albert-char-base-v2")
        model = AutoModel.from_pretrained("kb-albert-char-base-v2")
    elif args.model_name == 'koelectra':
        tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")

    model.load_state_dict(torch.load(args.ckpt_path)['state_dict'])
    model.to(device)
    model.eval()

    number_classifier = NumberClassification(device, tokenizer, args)

    test_dataset = NumClassDataset(dirpath=args.data_path,
                                    mode='test',
                                    tokenizer=tokenizer,
                                    args=args) 
    
    test_dataloader = DataLoader(test_dataset,
                                  num_workers=0,
                                  batch_size=args.batch_size,
                                  shuffle=False)
    
    test_acc = []
    list_acc = []
    correct_acc = []
    test_f1 = []
    result = []
    true_labels = []
    article_num = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            start = time.time()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'])
            
            last_hidden = outputs.last_hidden_state
            output, candidate_num_list, candidate_span_list, number_span_list = number_classifier(last_hidden, batch['input_ids'], batch['numbers'])

            target, decoded_label_list = get_label_class(tokenizer, batch['labels'], candidate_num_list)
            # target = torch.LongTensor(target).to(device)
            target = torch.LongTensor(target)
            preds = [torch.argmax(i).item() for i in output]
            
            for tgt, pred in zip(target, preds):
                if tgt == pred:
                    list_acc.append(1)
                else:
                    list_acc.append(0)
            
            # preds = [torch.argmax(i).item() for i in output]
            acc = accuracy_score(target, preds)
            # f1 = f1_score(target.detach().cpu(), preds, average='macro')
            test_acc.append(acc)
            # test_f1.append(f1)
            
            for max_idx, span_list, num_span, ids, label in zip(preds, candidate_span_list, number_span_list, batch['input_ids'], decoded_label_list):
                span = span_list[max_idx] # 최대값으로 선택된 값의 스팬 선택
                txt_ids = ids[span[0]:span[-1]] # 해당 스팬에 해당하는 인덱스 추출
                num_txt_ids = ids[num_span[0]:num_span[-1]]
                decoded_text = tokenizer.decode(txt_ids)
                decoded_art_num = tokenizer.decode(num_txt_ids)
                result.append(decoded_text)
                article_num.append(decoded_art_num)
                if decoded_text == label:
                    correct_acc.append(1)
                else:
                    correct_acc.append(0)

            true_labels += decoded_label_list

    print(f'Test Accuracy Average : {np.mean(test_acc)}')
    print(f'Average of list accuracy : {sum(list_acc)/len(list_acc)}')
    # print(f'Test F1 score Average : {np.mean(test_f1)}')
    # print(f'Length of Accuracy : {len(test_acc)}')
    print(f'Length of Labels : {len(true_labels)}')
    print(f'Length of Predict results : {len(result)}')
    print('==================Inference Finished!!!==================')

    df = pd.DataFrame({'article_num' : article_num,
                       'labels' : true_labels,
                       'results' : result,
                       'accuracy' : list_acc,
                       'correct_accuracy': correct_acc})
    
    epoch = args.ckpt_path.split('/')[-1].split('_')[0]
    # epoch = 17
    save_path = f'./data/output/{args.model_name}'
    make_directory(save_path)
    df.to_csv(f'{save_path}/result_v{args.version}_{epoch}.csv', index=False, encoding='utf-8')