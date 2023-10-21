import os
import pickle
import ndjson
import torch
import pandas as pd

from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def main():
    anno_path = "/data/dir_howto100m/anno"
    full_anno = pd.read_csv(os.path.join(anno_path, "HowTo100M_v1.csv"))
    full_anno = full_anno.dropna()

    category_candidates = [
        "Food and Entertaining",
        "Home and Garden",
        "Hobbies and Crafts",
    ]

    with open(os.path.join(anno_path, "HowToCaption.pickle"), "rb") as f:
        captions = pickle.load(f)

    anno_per_category = {}
    feat_path = "/data/dir_howto100m/feats/s3d/howto100m_s3d_features"

    for cname in category_candidates:
        anno_per_category[cname] = []
        anno = full_anno[(full_anno["category_1"] == cname) & (full_anno["rank"] == 1)]
        for row in anno.iterrows():
            _, data = row
            video_id = data["video_id"]
            c1 = data["category_1"]
            c2 = data["category_2"]

            if not os.path.exists(os.path.join(feat_path, f"{video_id}.npy")): 
                continue
            if video_id not in captions:
                continue

            anno_per_category[cname].append(
                {
                    "video_id": video_id,
                    "category_1": c1,
                    "category_2": c2,
                }
            )

    train_anno, val_anno, test_anno = [], [], []
    for cname, data in anno_per_category.items():
        n = len(data)
        cut1 = int(n * 0.7)
        cut2 = int(n * 0.8)
        train_anno.extend(data[:cut1])
        val_anno.extend(data[cut1:cut2])
        test_anno.extend(data[cut2:])

    id2cemb = {}
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    for anno in tqdm(train_anno + val_anno + test_anno, desc="[id2cemb]"):
        vid = anno["video_id"]
        text  = sum(captions[vid]["text"], [])
        with torch.no_grad():
            cemb = sbert.encode(text)
        id2cemb[vid] = cemb

    with open("anno/howto100m/train.ndjson", "w") as f:
        ndjson.dump(train_anno, f)
    with open("anno/howto100m/val.ndjson", "w") as f:
        ndjson.dump(val_anno, f)
    with open("anno/howto100m/test.ndjson", "w") as f:
        ndjson.dump(test_anno, f)

    torch.save(id2cemb, "anno/howto100m/id2cemb.pt")


if __name__ == "__main__":
    main()    