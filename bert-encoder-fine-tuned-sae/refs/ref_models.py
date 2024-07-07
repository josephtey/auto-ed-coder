import os
import torch
import torch.nn as nn

import transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification as AutoClassifier,
)
from transformers.models.deberta.modeling_deberta import StableDropout

transformers.logging.set_verbosity_error()

from canongrade.scripts.torch.data import SCALE_TOKENS
from canongrade.scripts.torch.utils import OrdinalLogisticModel


class RubricEncoder(nn.Module):
    def __init__(self, hidden_dim, dropout_prob=0.1):
        super().__init__()
        self.emb = nn.Embedding(len(SCALE_TOKENS), hidden_dim)
        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=dropout_prob,
            bidirectional=True,
        )
        self.ffn = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, inp):
        out = self.emb(inp)
        all_hidden_states, _ = self.rnn(out)
        out = all_hidden_states[:, 0]
        out = self.ffn(out)
        return out


class RubricModel(nn.Module):
    def __init__(
        self,
        encoder,
        num_rubrics,
        num_labels,
        hidden_dim=16,
        dropout_prob=0.1,
        **kwargs,
    ):
        super().__init__()
        if encoder in ["rnn", "fcn"]:
            self.emb = RubricEncoder(hidden_dim, dropout_prob=dropout_prob)
        else:
            self.emb = nn.Embedding(len(SCALE_TOKENS), 1)

        if encoder == "ordlr":
            self.classifier = OrdinalLogisticModel(
                predictor=nn.Linear(num_rubrics, 1), num_classes=num_labels
            )
        elif encoder == "lr":
            self.classifier = nn.Sequential(
                nn.Linear(num_rubrics, num_labels), nn.Dropout(p=dropout_prob)
            )
        elif encoder == "fcn":
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_prob),
                nn.Linear(num_rubrics, num_rubrics),
                nn.GELU(),
                nn.Dropout(p=dropout_prob),
                nn.Linear(num_rubrics, num_rubrics),
                nn.GELU(),
                nn.Dropout(p=dropout_prob),
                nn.Linear(num_rubrics, num_labels),
            )
        elif encoder == "rnn":
            self.classifier = nn.Sequential(
                StableDropout(dropout_prob),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                StableDropout(dropout_prob),
                nn.Linear(hidden_dim, num_labels),
            )

    def forward(self, batch):
        emb = self.emb(batch["feat_toks"]).squeeze(-1)
        logit = self.classifier(emb)
        return logit


class TransformerModel(nn.Module):
    def __init__(
        self,
        llm,
        num_labels,
        rubric_feat_dim=64,
        hidden_dim=32,
        num_rubrics=0,
        dropout_prob=0.1,
        **kwargs,
    ):
        super(TransformerModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(llm)
        # self.model = AutoClassifier.from_pretrained(
        self.model = AutoModel.from_pretrained(llm, num_labels=num_labels)
        self.device = "cpu"
        self.num_rubrics = num_rubrics
        self.use_rubric_features = num_rubrics != 0

        # classification head
        input_dim = self.model.config.hidden_size
        if self.use_rubric_features:
            self.rubric_encoder = RubricEncoder(
                rubric_feat_dim, dropout_prob=dropout_prob
            )
            input_dim += rubric_feat_dim

        print(f"use rubric features? {self.use_rubric_features}")
        self.classifier = nn.Sequential(
            StableDropout(dropout_prob),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            StableDropout(dropout_prob),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, batch):
        input_texts = batch["text"]
        inputs = self.tokenizer(
            input_texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = inputs.to(self.device)
        cls_inp = self.model(**inputs)
        # outputs = cls_inp.logits
        cls_inp = cls_inp.last_hidden_state[:, 0]

        if self.use_rubric_features:
            rubric_feat = self.rubric_encoder(batch["feat_toks"])
            cls_inp = torch.cat([cls_inp, rubric_feat], dim=1)
        outputs = self.classifier(cls_inp)

        return outputs

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def save(self, save_dir, **kwargs):
        ckpt_path = os.path.join(save_dir, "best.ckpt")
        torch.save({"state_dict": self.state_dict(), **kwargs}, ckpt_path)
        self.tokenizer.save_pretrained(save_dir)

    def load(self, save_dir):
        ckpt_path = os.path.join(save_dir, "best.ckpt")
        ckpt = torch.load(ckpt_path)
        self.load_state_dict(ckpt["state_dict"])
        self.tokenizer.from_pretrained(save_dir)
