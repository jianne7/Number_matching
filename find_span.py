import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
# from utils import *

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from konlpy.tag import Kkma


class FindNumber(nn.Module):
    def __init__(self):
        super().__init__()
        self.kkma  = Kkma()

    def forward(self, text):      
        date_num = self.date_number(text)
        money_num = self.money_number(text)
        etc_num = self.etc_number(text)
        re_num = list(set(date_num + money_num + etc_num))
        pos_num = self.pos_tagging(text, re_num)
        result = re_num+pos_num

        return result

    def date_number(self, text):
        # p = re.compile(r'\d분기|\d{2,4}년\s?\d분기|\d{2,4}.\d분기|\d{2,4}.\dQ|\dQ|Q\d|\d{2,4}년 \d{1,2}월 \d{1,2}일|\d{1,2}월 \d{1,2}일|\d{1,2}일|\d{2,4}년|\d{4}\.\d{1,2}\.\d{1,2}|\d+개월')
        p = re.compile(r'\d분기|\d{2,4}.\d분기|\d{2,4}.\dQ|\dQ|Q\d|\d{2,4}년 \d{1,2}월 \d{1,2}일|\d{1,2}월 \d{1,2}일|\d{1,2}일|\d{2,4}년|\d{4}\.\d{1,2}\.\d{1,2}|\d+개월')
        date_num = p.findall(text)
        return date_num
    
    def money_number(self, text):
        p = re.compile(r'\d?조?\s?\d?,?\d{1,4}억\s?\d?,?\d{1,4}만\s?\d?,?\d{1,4}원|\d?,?\d+조\s?\d?,?\d+억\s?\d?,?\d+천?만|\d?,?\d+억\s?\d?,?\d+천?만|\d?,?\d+억\s?\d?,?\d+천?만\s?\d+|\d?,?\d+억\s?\d?,?\d+백?\s?\d+만|\d+조\s?\d+억|\d?,?\d+억|\d+?,?\d+원|흑자전환|적자전환')
        money_num = p.findall(text)
        return money_num
    
    def etc_number(self, text):
        p = re.compile(r'\d+-\d+-\d+|\(명\)\n\d') #\d+,\d+|\d+,\d+,\d+|\d+\.\d+
        # p = re.compile(r'\d+-\d+-\d+|\(명\)\n\d|\d+,\d+|\d+,\d+,\d+')
        etc_num = p.findall(text)
        return etc_num
    
    def pos_tagging(self, text, re_num_list):
        num_list = []
        for pos in self.kkma.pos(text):
            if pos[1] == 'NR':
                num_list.append(pos[0])

        # 정규표현식에 있는 중복된 숫자표현 제거
        pos_num_list = []
        for num in list(set(num_list)):
            duplicate = [num in re_num for re_num in re_num_list]
            if True not in duplicate:
                pos_num_list.append(num)
        return pos_num_list
    

def remove_space(text):
    text = re.sub('(?<=\d)\. (?=\d)', '.', text)
    text = re.sub('(?<=\d), (?=\d)', ',', text)
    text = re.sub('(?<=\d) - (?=\d)', '-', text)

    return text


def find_span_index(text, tok_text, tokens_ids, target_num, ids_gap=None):
    text_s_idx = text.find(target_num)
    text_e_idx = text_s_idx + len(target_num)

    for i in range(len(tok_text)):
        if tok_text[i][1][0] <= text_s_idx <= tok_text[i][1][1]:
            word_start = i
        if tok_text[i][1][0] <= text_e_idx <= tok_text[i][1][1]:
            word_end = i
            break

    word_id = [w for w in range(word_start, word_end+1)]
    token_range = [i for i in range(len(tokens_ids)) if tokens_ids[i] in word_id]
  
    if ids_gap is None:
        ids_gap = 0

    start = token_range[0] + ids_gap
    end = token_range[-1] + 1 + ids_gap

    return (start, end)

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


if __name__ == '__main__':
    model_name = 'roberta'

    if model_name == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base") 
    elif model_name == 'kb-albert':
        tokenizer = AutoTokenizer.from_pretrained("kb-albert-char-base-v2")
    elif model_name == 'koelcetra':
        tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

    # data = pd.read_csv('./data/raw/total.csv')
    train_path = '/home/ujinne/project/number_classification/data/preprocessed/train.csv'
    valid_path = '/home/ujinne/project/number_classification/data/preprocessed/valid.csv'
    test_path = '/home/ujinne/project/number_classification/data/preprocessed/test.csv'

    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)
    test = pd.read_csv(test_path)

    data = pd.concat([train, valid, test], axis=0)

    find_num = FindNumber()

    no_label = []
    num_candi = []
    no_article_span = []
    for i in tqdm(range(len(data))):
        # text = data['filing_content'].iloc[i]
        truncated_filing = tokenizer.decode(tokenizer.encode(data['filing_content'].iloc[i])[:400], skip_special_tokens=True)
        truncated_article  = tokenizer.decode(tokenizer.encode(data['generated_article'].iloc[i]), skip_special_tokens=True)
        text = truncated_filing + '[SEP]' + truncated_article
        input_ids = torch.LongTensor(tokenizer(text, max_length=512, truncation=True)['input_ids'])

        text = tokenizer.decode(input_ids[1:-1])
        truncated_filing = text.split('[SEP]')[0]
        truncated_article = text.split('[SEP]')[1]

        tok_text = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        token_word_ids = tokenizer(text).word_ids()

        ids_gap = list(input_ids).index(tokenizer.sep_token_id) # 생성기사가 시작되기 전 SEP 토큰 ID 인덱스 위치
        tok_text_art = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(truncated_article)
        token_word_ids_art = tokenizer(truncated_article).word_ids()
        
        number = tokenizer.decode(tokenizer.encode(str(re.sub(r'\'', '',data['generated_numbers'].iloc[i]))), skip_special_tokens=True)

        label = re.sub('%', '', data['labels'].iloc[i])
        if label == '답없음':
            label = '[SEP]'
        else:
            label = tokenizer.decode(tokenizer.encode(label),  skip_special_tokens=True)

        filing = remove_space(truncated_filing)
        candidate_num = find_num(filing)
        candidate_num = [tokenizer.decode(tokenizer.encode(num), skip_special_tokens=True) for num in candidate_num] + ['[SEP]']
        # for num in candidate_num:
        #     if num == '[SEP]':
        #         (start, end) = (ids_gap, ids_gap+1)
        #     else:
        #         num = tokenizer.decode(tokenizer.encode(num), skip_special_tokens=True)
        #         (start, end) = find_span_index(text, tok_text, token_word_ids, num)
        try:
            candidate_num.index(label)
        except:
            no_label.append(i)

        try:
            (start, end) = find_span_index(truncated_article, tok_text_art, token_word_ids_art, number, ids_gap)
        except:
            no_article_span.append(i)        

    print(f'Model Name: {model_name}')
    print(f'No Label in Candidate Numbers: {len(no_label)}')
    print(f'No Number span in Article: {len(no_article_span)}')
    print(f'total : {len(set(no_label+no_article_span))}')
    print(f'{len(data)} - {len(set(no_label+no_article_span))} = {len(data)-len(set(no_label+no_article_span))}')
    # print(f'Average label candidate class: {np.mean(num_candi):.4f}')
    # print(f'Max label candidate class: {max(num_candi)}')
    # print(f'Min label candidate class: {min(num_candi)}')
    #     try:
    #         candidate_num.index(label)
    #     except:
    #         no_label.append(i)

    # no_data = list(set(no_label+no_article_span))
    # df = data.drop(no_data)
    # make_directory(f'./data/v3/{model_name}')
    # df.to_csv(f'./data/v3/{model_name}/total.csv', index=False)