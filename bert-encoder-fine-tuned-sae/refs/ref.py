import argparse
import os
import json
from dotmap import DotMap
from tqdm import tqdm, trange
import numpy as np
from pprint import pprint
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    accuracy_score,
    pairwise,
    confusion_matrix,
)

from canongrade.const import ROOT_DIR
from canongrade.scripts.torch.data import ASAPDataset, batch_to_device
from canongrade.scripts.torch.models import TransformerModel, RubricModel

LLM = {
    "deberta": "microsoft/deberta-v3-base",
    "bert": "google-bert/bert-base-uncased",
    "electra": "google/electra-large-discriminator",
    "roberta": "FacebookAI/roberta-large",
}

METHOD = {
    "llm": TransformerModel,
    "llm+rubric": TransformerModel,
    "rubric": RubricModel,
}

FEAT_DIR = {
    "03": "asap_03_scale_desc",
    "04": "asap_04_scale_desc_multi",
    "07": "asap_07_scale_desc",
    "08": "asap_08_scale_desc_multi",
}

RUBRIC_DIR = {
    "03": "scale_desc_03",
    "04": "scale_desc_04",
    "07": "scale_desc_07",
    "08": "scale_desc_08",
}

RUBRIC_NAMES = {
    "03": ["rubric_1", "rubric_2", "rubric_3"],
    "04": ["new_rubric_1", "new_rubric_2", "new_rubric_3"],
    #'07': ['rubric_1', 'rubric_2', 'rubric_3'],
    "07": ["small_rubric_1", "small_rubric_2"],
    "08": ["new_rubric_1", "new_rubric_2"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dname", type=str)
    parser.add_argument("method", type=str)
    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--llm", type=str, default=None)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--n-each", type=int, default=None)
    parser.add_argument("--silent", action="store_true", default=False)
    parser.add_argument("--no-save", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    if args.method in ["rubric", "llm+rubric"] and args.dname not in FEAT_DIR:
        return

    device = "cpu" if args.device < 0 else f"cuda:{args.device}"
    save_name = f"asap{args.dname}_{args.method}"
    if args.method == "rubric":
        save_name += f"_enc-{args.encoder}"
    elif args.method in ["llm", "llm+rubric"]:
        save_name += f"_llm-{args.llm}"
    if args.n_each is not None:
        save_name += f"_neach{args.n_each}"
    save_dir = os.path.join(ROOT_DIR, "scripts", "out", "torch", save_name)

    if os.path.isfile(os.path.join(save_dir, "perf.json")) and not args.no_save:
        if not args.overwrite:
            print(f"{save_dir} run already exists.")
            return
        print(f"removing directory: {save_dir}")
        shutil.rmtree(save_dir)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    batch_size = 12 if "llm" in args.method else 128
    learning_rate = 3e-5 if "llm" in args.method else 1e-3
    epochs = 100 if args.method in ["llm", "llm+rubric"] else 300

    config = {
        "data": {"dname": args.dname, "n_each": args.n_each, "split_name": "small_cv1"},
        "model": {
            "llm": (LLM[args.llm] if args.llm in LLM else None),
            "encoder": args.encoder,
            "hidden_dim": 32,
        },
        "optim": {"lr": learning_rate},
        "method": args.method,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": 8,
    }
    config = DotMap(config)

    if args.method in ["llm+rubric", "rubric"]:
        config.data.update(
            {
                "feat_dir": FEAT_DIR[args.dname],
                "rubric_dir": RUBRIC_DIR[args.dname],
                "rubric_names": RUBRIC_NAMES[args.dname],
            }
        )

    if args.method == "llm+rubric":
        config.model.rubric_feat_dim = 64

    pprint(config.toDict())

    train_dataset = ASAPDataset(split="train", **config.data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )

    valid_dataset = ASAPDataset(split="valid", **config.data)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )

    config.model.num_rubrics = len(train_dataset.rubrics)
    config.model.num_labels = train_dataset.num_labels

    model = METHOD[config.method](**config.model).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), **config.optim)

    best_qwk = -float("inf")
    best_acc = None
    for epoch in (trange if args.silent else range)(config.epochs):
        if not args.silent:
            print(f"==== epoch {epoch+1} ====")
        model.train()
        total_loss = 0.0
        for batch in train_loader if args.silent else tqdm(train_loader):
            batch = batch_to_device(batch, args.device)

            optimizer.zero_grad()
            pred_logits = model(batch)
            true = batch["score"]
            loss = criterion(pred_logits, true)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if not args.silent:
            print(f"train loss: {total_loss}")

        pred_lst, true_lst = [], []
        model.eval()
        for batch in valid_loader if args.silent else tqdm(valid_loader):
            batch = batch_to_device(batch, args.device)
            with torch.no_grad():
                pred_logits = model(batch)
            pred = torch.argmax(pred_logits, dim=1)

            pred_lst.append(pred.cpu().detach().numpy())
            true_lst.append(batch["score"].cpu().numpy())

        pred = np.concatenate(pred_lst)
        true = np.concatenate(true_lst)

        qwk = cohen_kappa_score(pred, true, weights="quadratic")
        acc = accuracy_score(pred, true)
        if qwk > best_qwk:
            best_qwk = qwk
            best_acc = acc
            if not args.no_save:
                model.save(save_dir, config=config)
        if not args.silent:
            print(
                f"curr qwk: {qwk:.3f}\tbest qwk: {best_qwk:.3f}\tbest acc: {best_acc:.3f}"
            )

        if not args.no_save:
            with open(os.path.join(save_dir, "perf.json"), "w") as f:
                json.dump({"qwk": best_qwk, "acc": best_acc}, f, indent=4)


if __name__ == "__main__":
    main()
