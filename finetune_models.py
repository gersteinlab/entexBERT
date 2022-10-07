import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss, BCEWithLogitsLoss
from torch.autograd import Variable

from transformers import (BertConfig, BertModel, BertPreTrainedModel, BertForSequenceClassification)


####### Functionalities
class FocalLoss(nn.Module):
    '''
    Focal loss (https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py)
    '''
    

####### Models
class BertForSequenceClassificationFL(BertModel):
    """
    Use focal loss to address imbalanced datasets.
    Config:
    - alpha, gamma: parameters for focal loss
    """
    def __init__(self, config):
        super().__init__(config)      
        self.num_labels = config.num_labels
        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.aggregation = nn.Conv1d(in_channels=config.k, out_channels=1, kernel_size=1)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.alpha = config.alpha
        self.gamma = config.gamma
        
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        epi_vec = None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(torch.cat((pooled_output, epi_vec), dim=1))
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    
    
class BertForSNPClassification(BertModel):
    """
    BERT model that is built on the central token embedding, instead of [CLS].
    More specifically, the classifier is built on the average of all tokens that cover the central nucleotide (i.e. the location of the SNP)
    The model requires the k-mer length argument to determine which tokens to look at.
    """
    def __init__(self, config):
        super().__init__(config)
        self.k = int(config.k)
        
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.aggregation = nn.Conv1d(in_channels=config.k, out_channels=1, kernel_size=1)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
    
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        
        
        sequence_output = outputs[0]
        d = int(self.k /2)
        sequence_output = sequence_output[:,(127-d):(127+d),:]
        
        pooled_output = sequence_output.mean(axis=1)
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


### Sub-BERTs: use only lower layer for the classification
class SubBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.pred_layer = config.pred_layer
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
    
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        
        pooled_output = outputs[2][self.pred_layer][:,0,:] ## TXL: difference from the original BertForSequenceClassification

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    
class SubBertForSNPClassification(BertModel):
    """
    BERT model that is built on the central token embedding, instead of [CLS].
    More specifically, the classifier is built on the average of all tokens that cover the central nucleotide (i.e. the location of the SNP)
    The model requires the k-mer length argument to determine which tokens to look at.
    """
    def __init__(self, config):
        super().__init__(config)
        self.k = int(config.k)
        self.pred_layer = config.pred_layer
        
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.aggregation = nn.Conv1d(in_channels=config.k, out_channels=1, kernel_size=1)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
    
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        
        
        sequence_output = outputs[2][self.pred_layer]
        d = int(self.k /2)
        sequence_output = sequence_output[:,(127-d):(127+d),:]
        
        pooled_output = sequence_output.mean(axis=1)
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    
    
    
##########
class BertForMultilabelSequenceClassification(BertForSequenceClassification):
    """
    BERT model with linear layer for multi-label classification
    """
    def __init__(self, config, num_hidden_layer=768):
        super().__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        self.num_hidden_layer = num_hidden_layer
        
        '''
        ### One-layer model
        self.dropout = nn.Dropout(config.hidden_dropout_prob)        
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        '''
        
        ### Two-layer model
        self.dropout0 = nn.Dropout(config.hidden_dropout_prob)        
        self.classifier0 = nn.Linear(config.hidden_size, num_hidden_layer)
        self.relu = nn.ReLU()
        
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)        
        self.classifier1 = nn.Linear(num_hidden_layer, self.num_labels)
        #self.sigmoid = nn.Sigmoid()
        
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)

        pooled_output = outputs[1]
        
        '''
        ### One-layer model
        pooled_output = self.dropout(pooled_output)        
        logits = self.classifier(pooled_output)
        '''
        
        ### Two-layer model
        pooled_output = self.dropout0(pooled_output)        
        classifier0_output = self.classifier0(pooled_output)
        classifier0_output = self.relu(classifier0_output)
        
        classifier0_output = self.dropout1(classifier0_output)        
        logits = self.classifier1(classifier0_output)
        
        #output = self.sigmoid(logits)
        output = logits
        outputs = (output,) + outputs[2:]
        
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(output.view(-1), labels.view(-1))
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(output.view(self.num_labels,-1), labels.view(self.num_labels,-1))
            outputs = (loss,) + outputs
        
        return outputs

    
    