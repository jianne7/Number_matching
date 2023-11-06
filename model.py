import re
import torch
import torch.nn as nn

from transformers import BertModel, AutoModel


class Basemodel(nn.Module):
    def __init__(self, args, device, tokenizer):
        super().__init__()
        if args.model == 'bert':
            self.model = BertModel.from_pretrained('skt/kobert-base-v1')
        elif args.model == 'roberta':
            self.model = AutoModel.from_pretrained("klue/roberta-base")
        self.device = device
        self.tokenizer = tokenizer
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids = input_ids, attention_mask = attention_mask)
        last_hidden = outputs.last_hidden_state

        return last_hidden