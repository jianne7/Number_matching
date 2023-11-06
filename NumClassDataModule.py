import re
import numpy as np
import pandas as pd

import torch
import torch.utils.data
from torch.utils.data import Dataset

class NumClassDataset(Dataset):
    def __init__(self, dirpath, mode, tokenizer, args):
        self.dirpath = dirpath
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.args = args

        data_path = f'{self.dirpath}/{args.model_name}/{self.mode}.csv'
        self.data = pd.read_csv(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        filing = self.data['filing_content'].iloc[index]
        article = self.data['generated_article'].iloc[index]
        number = str(re.sub(r"\'", "", self.data['generated_numbers'].iloc[index]))

        if self.args.mask == True:
            article = re.sub(number, self.tokenizer.mask_token, article)
            truncated_filing = self.tokenizer.decode(self.tokenizer.encode(filing)[:400][1:-1]) + '[SEP]'
            truncated_article = self.tokenizer.decode(self.tokenizer.encode(article)[1:-1])
        else:
            truncated_filing = self.tokenizer.decode(self.tokenizer.encode(filing)[:400], skip_special_tokens=True) + '[SEP]'
            truncated_article = self.tokenizer.decode(self.tokenizer.encode(article), skip_special_tokens=True)

        input_text = truncated_filing + truncated_article
        inputs = self.tokenizer(input_text, max_length=self.max_len, truncation=True)

        input_ids = self.add_padding_data(inputs['input_ids'])
        input_ids = torch.from_numpy(np.array(input_ids, dtype=np.int_))

        attn_mask = self.add_padding_data(inputs['attention_mask'])
        attn_mask = torch.from_numpy(np.array(attn_mask, dtype=np.int_))

        src_num = self.tokenizer.encode(number)
        src_num = self.add_ignored_data(src_num, 256)
        src_num = torch.from_numpy(np.array(src_num, dtype=np.int_))
        
        label = re.sub('%', '', self.data['labels'].iloc[index])
        if label == '답없음':
            label = '[SEP]'

        tgt_num = self.tokenizer.encode(label)
        tgt_num = self.add_ignored_data(tgt_num, 256)
        tgt_num = torch.from_numpy(np.array(tgt_num, dtype=np.int_))

        return {
                'input_ids' : input_ids,
                'attention_mask' : attn_mask,
                'numbers' : src_num,
                'labels' : tgt_num
            }

    def add_padding_data(self, inputs) :
        if len(inputs) < self.max_len :
            pad = np.array([self.tokenizer.pad_token_id] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else :
            inputs = inputs[:self.max_len]
        return inputs
    
    def add_ignored_data(self, inputs, max_len) :
        if len(inputs) < max_len :
            pad = np.array([-100]*(max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else :
            inputs = inputs[:max_len]
        return inputs