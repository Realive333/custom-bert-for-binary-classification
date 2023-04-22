import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

class AveragedBERT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.aggregate = nn.Linear(in_features=768, out_features=100)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=100, out_features=2)
        self.loss = nn.CrossEntropyLoss()

        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        input_ids = input_ids.squeeze()
        attention_mask = attention_mask.squeeze()
        token_type_ids = token_type_ids.squeeze()
        
        outputs = self.bert(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            **kwargs
        )
        
        pooled_outputs = outputs.pooler_output
        
        aggregate_outputs = self.aggregate(pooled_outputs)
        relu_outputs = self.relu(aggregate_outputs)
        avg_output = torch.mean(relu_outputs, 0, True) # True: remain the original shape
        cls_output = self.linear(avg_output)
        
        loss = None
        if labels is not None:
            loss = self.loss(cls_output, labels)
    
        return SequenceClassifierOutput(
            loss = loss,
            logits = cls_output,
            hidden_states = outputs.hidden_states,
            attentions = outputs.attentions
        )  
    
class ConcatedBERT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.linear = nn.Linear(in_features=768*5, out_features=2)
        self.loss = nn.CrossEntropyLoss()

        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        input_ids = input_ids.squeeze()
        attention_mask = attention_mask.squeeze()
        token_type_ids = token_type_ids.squeeze()
        
        outputs = self.bert(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            **kwargs
        )
        
        pooled_outputs = outputs.pooler_output
        cat_output = pooled_outputs.view(1, -1)
        cls_output = self.linear(cat_output)
        
        loss = None
        if labels is not None:
            loss = self.loss(cls_output, labels)
    
        return SequenceClassifierOutput(
            loss = loss,
            logits = cls_output,
            hidden_states = outputs.hidden_states,
            attentions = outputs.attentions
        )  
    
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'acc': acc, 'f1': f1}