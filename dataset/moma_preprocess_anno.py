import ndjson
import torch

from momaapi import MOMA
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def main():
    moma = MOMA("/data/dir_moma")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    
    id2cemb = {}
    for split in ["train", "val", "test"]:
        anno = []
        ids_act = moma.get_ids_act(split=split)
        for act in tqdm(moma.get_anns_act(ids_act=ids_act), desc=f"[{split}]"):
            if act.id == "1YzGUyM3P2k": # issue: no graph
                continue
            
            anno.append(
                {
                    "video_id": act.id,
                    "cname": act.cname,
                }
            )

            captions = []
            for sact in moma.get_anns_sact(ids_sact=act.ids_sact):
                captions.append(sact.cname)
            
            with torch.no_grad():
                cemb = sbert.encode(captions)
                cemb = torch.from_numpy(cemb).float()
            id2cemb[act.id] = cemb

        with open(f"anno/moma/{split}.ndjson", "w") as f:
            ndjson.dump(anno, f)

    torch.save(id2cemb, f"anno/moma/id2cemb.pt")


if __name__ == "__main__":
    main()