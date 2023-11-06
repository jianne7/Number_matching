import re

import torch
import torch.nn as nn
from konlpy.tag import Kkma


class FindNumber(nn.Module):
    def __init__(self):
        super().__init__()
        self.kkma  = Kkma()

    def forward(self, text):
        # text = self.remove_space(text)
        
        date_num = self.date_number(text)
        money_num = self.money_number(text)
        etc_num = self.etc_number(text)
        re_num = list(set(date_num + money_num + etc_num))
        pos_num = self.pos_tagging(text, re_num)
        result = re_num+pos_num

        return result
    
    # def remove_space(self, text):
    #     text = re.sub('(?<=\d)\. (?=\d)', '.', text)
    #     text = re.sub('(?<=\d), (?=\d)', ',', text)
    #     text = re.sub('(?<=\d) - (?=\d)', '-', text)

    #     return text

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
        p = re.compile(r'\d+-\d+-\d+|\(명\)\n\d|\d+,\d+|\d+,\d+,\d+|\d+\.\d+') #\d+\.\d+
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
            duplicate = [num in re_num for re_num in re_num_list] # [num in re_num for re_num in re_num_list] 고민좀..
            if True not in duplicate:
                pos_num_list.append(num)
        return pos_num_list


class NumberClassification(nn.Module):
    def __init__(self, device, tokenizer, args):
        super().__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.find_num = FindNumber()
        self.args = args

    def forward(self, last_hidden, input_ids, numbers):
        candidate_num_list, candidate_span_list, number_span_list = self.get_span_idx(input_ids, numbers)     
        candidate_hidden_state = self.get_span_candidate_hidden_state(last_hidden, candidate_span_list) # 숫자 후보에 해당하는 hidden state 추출 (batch, max_class_num, hidden_size)
        number_hidden_state = self.get_span_number_hidden_state(last_hidden, number_span_list) # 생성기사 숫자에 해당하는 hidden state 추출 (batch, 1, hidden)

        output = torch.matmul(number_hidden_state, candidate_hidden_state.transpose(1, -1)).squeeze(dim=1)

        return output, candidate_num_list, candidate_span_list, number_span_list
    

    def get_span_idx(self, input_ids, numbers):
        '''
        각 숫자 표현의 위치에 해당하는 span index 추출
        '''
        candidate_num_list = []
        candidate_span_list = []
        number_span_list = []
        for input_id, number in zip(input_ids, numbers):
            text = self.tokenizer.decode(input_id[1:-1]) # [CLS] 제거한 텍스트
            filing = text.split('[SEP]')[0]
            article = text.split('[SEP]')[1]

            tok_text = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            token_word_ids = self.tokenizer(text).word_ids()

            ids_gap = list(input_id).index(self.tokenizer.sep_token_id) # 생성기사가 시작되기 전 SEP 토큰 ID 인덱스 위치
            tok_text_art = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(article)
            token_word_ids_art = self.tokenizer(article).word_ids()

            # remove -100 padding
            pad_ind = torch.where(number == -100)[0][0]
            number = number[:pad_ind][1:-1]
            number = self.tokenizer.decode(number) # for except [CLS], and last [SEP]
            
            filing = self.remove_space(filing)
            candidate_num = self.find_num(filing)
            candidate_num = [self.tokenizer.decode(self.tokenizer.encode(num), skip_special_tokens=True) for num in candidate_num] + ['[SEP]']
            candidate_span = self.get_candidate_span(candidate_num, text, tok_text, token_word_ids, ids_gap) # 공시문서 내의 각 숫자 표현의 위치
            
            if self.args.mask == True:
                mask_id = list(input_id).index(self.tokenizer.mask_token_id)
                number_span = (mask_id, mask_id+1)
            else:
                number_span = self.find_index_range(article, tok_text_art, token_word_ids_art, number, ids_gap) # 생성기사 내의 숫자 표현 위치

            candidate_num_list.append(candidate_num)
            candidate_span_list.append(candidate_span)
            number_span_list.append(number_span)

        return candidate_num_list, candidate_span_list, number_span_list

    def remove_space(self, text):
        text = re.sub('(?<=\d)\. (?=\d)', '.', text)
        text = re.sub('(?<=\d), (?=\d)', ',', text)
        text = re.sub('(?<=\d) - (?=\d)', '-', text)

        return text

    def get_span_candidate_hidden_state(self, last_hidden_state, span_list: list):
        batch_hidden = []
        for hidden, span in zip(last_hidden_state, span_list):
            centroid = []
            for s in span:
                centroid.append(torch.mean(hidden[s[0]:s[1]], dim=0))
            centroid = torch.stack(centroid)
            batch_hidden.append(centroid)
        
        result = []
        max_len_labels = max(list(map(lambda x: len(x), batch_hidden)))
        for hidden in batch_hidden:
            add_size = max_len_labels - (hidden.size(0))
            if add_size != 0:
                # add_tensor = torch.tensor([-100]*hidden.size(-1))
                add_tensor = torch.tensor([0]*hidden.size(-1))
                add_stack  = torch.stack([add_tensor]*add_size).to(self.device)
                result.append(torch.concat([hidden, add_stack], dim=0))
            else:
                result.append(hidden)
        result = torch.stack(result)

        return result

    def get_span_number_hidden_state(self, last_hidden_state, span_list: list):
        batch_hidden = []
        for hidden, span in zip(last_hidden_state, span_list):
            centroid = torch.mean(hidden[span[0]:span[1]], dim=0)
            batch_hidden.append(centroid)
        result = torch.stack(batch_hidden).unsqueeze(dim=1)

        return result


    def get_candidate_span(self, candidate_num, text, tok_text, token_word_ids, ids_gap):
        '''
        후보 숫자들을 인코딩하여 해당 숫자들이 input값의 위치를 span으로 찾아줌
        '''
        candidate_span = []
        for num in candidate_num:
            if num == '[SEP]':
                span = (ids_gap, ids_gap+1)
            else:
                span = self.find_index_range(text, tok_text, token_word_ids, num)
            candidate_span.append(span)

        return candidate_span


    def find_index_range(self, text, tok_text, tokens_ids, target_num, ids_gap=None):
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