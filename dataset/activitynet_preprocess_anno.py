import os
import json
import ndjson
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def main():
    id2cemb = {}
    anno_path = "/data/dir_activitynet/anno"
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    for split in ["train", "val_1"]:
        with open(os.path.join(anno_path, f"{split}.json"), "r") as f:
            data = json.load(f)
        
        if split == "train": # issue: video file corrupted
            data.pop("v_N8otQdjR96s")

        video_ids = []
        for vid, captions in tqdm(data.items(), desc=f"[{split}]"):
            video_ids.append(vid)
            sentences = captions["sentences"]
            sentences = [s.strip() for s in sentences]
            with torch.no_grad():
                cemb = sbert.encode(sentences)
            id2cemb[vid] = cemb

        with open(f"anno/activitynet/{split}.json", "w") as f:
            json.dump(video_ids, f)

    torch.save(id2cemb, "anno/activitynet/id2cemb.pt")


if __name__ == "__main__":
    main()