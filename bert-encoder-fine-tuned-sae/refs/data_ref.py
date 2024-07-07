import json
import os
import pandas as pd
import pickle

import torch
from torch.utils.data import Dataset
from canongrade.const import DATA_DIR, ROOT_DIR

RUBRIC_DIR = os.path.join(ROOT_DIR, "scripts", "rubrics")
FEAT_DIR = os.path.join(ROOT_DIR, "scripts", "out")

SCALE_DICT = {
    "Strongly Agree": "__strongly_agree__",
    "Agree": "__agree__",
    "Neutral": "__neutral__",
    "Disagree": "__disagree__",
    "Strongly Disagree": "__disagree__",
}
SCALE_TOKENS = {k: i for i, k in enumerate(set(SCALE_DICT.values()))}


def batch_to_device(batch, device):
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }
    return batch


class ASAPDataset(Dataset):
    def __init__(
        self,
        dname,
        split,
        split_name="cv1",
        n_each=None,
        agreed_only=False,
        feat_dir=None,
        rubric_dir=None,
        rubric_names=None,
    ):
        self.dname = dname
        path = os.path.join(DATA_DIR, "raw", "asap", "raw", f"df_{dname}.csv")
        split_fname = os.path.join(
            DATA_DIR, "processed", f"asap_{dname}", "splits", f"{split_name}.pkl"
        )
        with open(split_fname, "rb") as f:
            sid_splits = pickle.load(f)
        sid_list = sid_splits[split]

        rubrics = []
        features = {}
        if feat_dir is not None:
            rubric_dir = os.path.join(RUBRIC_DIR, rubric_dir)
            for r_name in rubric_names:
                fname = os.path.join(rubric_dir, f"{r_name}.json")
                with open(fname) as f:
                    rubric_lst = json.load(f)
                    rubrics.extend([r["rubric_item"] for r in rubric_lst])
            # remove duplicate rubrics
            rubrics = list(set(rubrics))
            print(f"{len(rubrics)} rubrics")

            for sid in sid_list:
                features[sid] = {}
                try:
                    for rname in rubric_names:
                        fname = os.path.join(FEAT_DIR, feat_dir, sid, rname, "out.json")
                        with open(fname) as f:
                            item = json.load(f)
                        features[sid].update(item)
                except FileNotFoundError:
                    fname = os.path.join(FEAT_DIR, feat_dir, sid, "out.json")
                    with open(fname) as f:
                        item = json.load(f)
                    features[sid] = item

        if split != "train":
            n_each = None

        self.rubrics = rubrics
        self.features = features
        df = pd.read_csv(path)
        self.num_labels = len(df.Score1.unique())
        df = df[df.Id.isin([int(sid) for sid in sid_list])]
        if n_each is not None:
            df = df.groupby("Score1").sample(n=n_each, random_state=1)
        self.df = df
        self.sid_list = df.Id.astype(str).tolist()

    def __len__(self):
        return len(self.sid_list)

    def __getitem__(self, index):
        df = self.df
        sid = self.sid_list[index]
        row = df[df.Id == int(sid)].iloc[0]
        feat_str = ""
        feat_tok = []
        for i, rubric in enumerate(self.rubrics):
            scale = SCALE_DICT[self.features[sid][rubric]]
            scale_tok = SCALE_TOKENS[scale]
            # feat_str += f'rubric_{i} {scale} '
            feat_str += f"{scale} "
            feat_tok.append(scale_tok)
        feat_str = feat_str.strip()
        item = {
            "idx": index,
            "sid": int(sid),
            "score": row.Score1,
            "text": row.EssayText,
            "feat_str": feat_str,
            "feat_toks": torch.LongTensor(feat_tok),
            "score2": row.Score2 if "Score2" in row else -1,
        }
        return item

    @staticmethod
    def collate_fn(batch):
        new_batch = {
            "idx": torch.LongTensor([b["idx"] for b in batch]),
            "sid": torch.LongTensor([b["sid"] for b in batch]),
            "text": [b["text"] for b in batch],
            "feat_str": [b["feat_str"] for b in batch],
            "feat_toks": torch.stack([b["feat_toks"] for b in batch]),
            "score": torch.LongTensor([b["score"] for b in batch]),
        }

        # either all entries have score 2 or none of them do
        if batch[0]["score2"] == -1:
            new_batch["score2"] = None
        else:
            score2_lst = [b["score2"] for b in batch]
            new_batch["score2"] = torch.LongTensor(score2_lst)

        return new_batch
