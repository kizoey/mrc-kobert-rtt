import random
import logging

import torch
import numpy as np

from bert_models.src.modeling_bert import BertForQuestionAnsweringAVPool, BertForQuestionAnswering
from bert_models.src import ElectraForQuestionAnswering

from bert_models.src import (
    KoBertTokenizer,
    HanBertTokenizer
)
from transformers import (
    BertConfig,
    DistilBertConfig,
    ElectraConfig,
    XLMRobertaConfig,
    BertTokenizer,
    ElectraTokenizer,
    XLMRobertaTokenizer,
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    ElectraForSequenceClassification,
    XLMRobertaForSequenceClassification,
    BertForTokenClassification,
    DistilBertForTokenClassification,
    ElectraForTokenClassification,
    XLMRobertaForTokenClassification,
    DistilBertForQuestionAnswering,
    XLMRobertaForQuestionAnswering,
)

CONFIG_CLASSES = {
    "kobert": BertConfig,
    "distilkobert": DistilBertConfig,
    "hanbert": BertConfig,
    "koelectra-base": ElectraConfig,
    "koelectra-small": ElectraConfig,
    "koelectra-base-v2": ElectraConfig,
    "koelectra-small-v2": ElectraConfig,
    "koelectra-base-v3": ElectraConfig,
    "koelectra-small-v3": ElectraConfig,
    "koelectra-base-v3-post": ElectraConfig,
    "xlm-roberta": XLMRobertaConfig,
    'kcbert': BertConfig,
    'kcbert-ifv': BertConfig,
    'kcelectra-base-v1': ElectraConfig,
}

TOKENIZER_CLASSES = {
    "kobert": KoBertTokenizer,
    "distilkobert": KoBertTokenizer,
    "hanbert": HanBertTokenizer,
    "koelectra-base": ElectraTokenizer,
    "koelectra-small": ElectraTokenizer,
    "koelectra-base-v2": ElectraTokenizer,
    "koelectra-base-v3": ElectraTokenizer,
    "koelectra-base-v3-post": ElectraTokenizer,
    "koelectra-small-v2": ElectraTokenizer,
    "koelectra-small-v3": ElectraTokenizer,
    "xlm-roberta": XLMRobertaTokenizer,
    'kcbert': BertTokenizer,
    'kcbert-ifv': BertTokenizer,
    "kcelectra-base-v1": ElectraTokenizer,
}

MODEL_FOR_SEQUENCE_CLASSIFICATION = {
    "kobert": BertForSequenceClassification,
    "distilkobert": DistilBertForSequenceClassification,
    "hanbert": BertForSequenceClassification,
    "koelectra-base": ElectraForSequenceClassification,
    "koelectra-small": ElectraForSequenceClassification,
    "koelectra-base-v2": ElectraForSequenceClassification,
    "koelectra-base-v3": ElectraForSequenceClassification,
    "koelectra-base-v3-post": ElectraForSequenceClassification,
    "koelectra-small-v2": ElectraForSequenceClassification,
    "koelectra-small-v3": ElectraForSequenceClassification,
    "xlm-roberta": XLMRobertaForSequenceClassification,
    'kcbert': BertForSequenceClassification,
    'kcbert-ifv': BertForSequenceClassification,
    "kcelectra-base-v1": ElectraForSequenceClassification,
}

MODEL_FOR_TOKEN_CLASSIFICATION = {
    "kobert": BertForTokenClassification,
    "distilkobert": DistilBertForTokenClassification,
    "hanbert": BertForTokenClassification,
    "koelectra-base": ElectraForTokenClassification,
    "koelectra-small": ElectraForTokenClassification,
    "koelectra-base-v2": ElectraForTokenClassification,
    "koelectra-base-v3": ElectraForTokenClassification,
    "koelectra-base-v3-post": ElectraForTokenClassification,
    "koelectra-small-v2": ElectraForTokenClassification,
    "koelectra-small-v3": ElectraForTokenClassification,
    "xlm-roberta": XLMRobertaForTokenClassification,
    'kcbert': BertForTokenClassification,
    'kcbert-ifv': BertForTokenClassification,
    "kcelectra-base-v1": ElectraForTokenClassification,
}

MODEL_FOR_QUESTION_ANSWERING = {
    "kobert": BertForQuestionAnswering,
    "distilkobert": DistilBertForQuestionAnswering,
    "hanbert": BertForQuestionAnswering,
    "koelectra-base": ElectraForQuestionAnswering,
    "koelectra-small": ElectraForQuestionAnswering,
    "koelectra-base-v2": ElectraForQuestionAnswering,
    "koelectra-base-v3": ElectraForQuestionAnswering,
    "koelectra-base-v3-post": ElectraForQuestionAnswering,
    "koelectra-small-v2": ElectraForQuestionAnswering,
    "koelectra-small-v3": ElectraForQuestionAnswering,
    "xlm-roberta": XLMRobertaForQuestionAnswering,
    'kcbert': BertForQuestionAnswering,
    'kcbert_ifv': BertForQuestionAnsweringAVPool,
    "kcelectra-base-v1": ElectraForQuestionAnswering,
}


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)