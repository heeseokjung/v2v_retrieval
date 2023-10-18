import os
import argparse
import ndjson
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from torchvision.ops import box_iou
from momaapi import MOMA
from sentence_transformers import SentenceTransformer
from  tqdm import tqdm


patches = []
for i in np.arange(0, 224-16+1, 16):
    for j in np.arange(0, 224-16+1, 16):
        patches.append((j, i, j+16, i+16))
patches = np.stack(patches, axis=0)
patches = torch.LongTensor(patches)


def get_caption_emb(model, captions):
    ...


def get_object_repr(feat, obj_list, id2eig):
    """
    args:
        - feat: n_clips x n_patches x d
                where the first patch is CLS token
        - obj_list: list of node which is dictionary of
            "hid": higher-order interaction id
            "eid": entity id
            "t": timestamp
            "nidx": nodeidx
            "bbox": bbox coordinate
        - id2eig: eigenvalues & eigenvectors of graph(spatial) Laplacian
    """

    feat = torch.from_numpy(feat[:, 1:, :]) # remove CLS token
    n_clips, n_patches, d = feat.shape
    feat = feat.reshape(n_clips * 4, -1, d) # n_total_frames x patches_per_frame x d

    global patches
    obj_bbox_list = [obj["bbox"] for obj in obj_list]
    obj_bbox_list = torch.LongTensor(obj_bbox_list) # n_total_objects x 4
    bbox_iou = box_iou(obj_bbox_list, patches) # n_total_objects x patches_per_frame

    x, pos_enc = [], []
    ids_hoi, ids_entity, ts_l = [], [], []
    for obj, iou in zip(obj_list, bbox_iou):
        if obj["t"] >= feat.shape[0]:
            continue
        frame_feat = feat[obj["t"],:,:] # patches_per_frame x d
        x.append(torch.matmul(F.softmax(iou, dim=0), frame_feat)) # d,
        u = id2eig[obj["hid"]]["u"]
        pos_enc.append(u[obj["nidx"],:])
        ids_hoi.append(obj["hid"])
        ids_entity.append(obj["eid"])
        ts_l.append(obj["t"])

    x = torch.stack(x, dim=0) # n_total_objects x d
    x = x.cpu().numpy()
    pos_enc = np.stack(pos_enc, axis=0) # n_total_objects x n_total_objects
    ts_l = np.array(ts_l, dtype=np.int64)

    return x, pos_enc, ids_hoi, ids_entity, ts_l


def get_laplacian_eig(A):
    """
    args:
        - A: adjacency matrix
    returns:
        - eigenvalues & eigenvectors of normalized graph Laplacian
    """
    
    deg = A.sum(axis=1)
    D = np.diag(deg.clip(1.) ** -0.5)
    L = np.eye(len(A)) - D @ A @ D # Normalized graph Laplacian

    e, u = np.linalg.eig(L) # eigenvalues, eigenvectors

    # sanity check
    for i in range(len(e)):
        x = L @ u[:, i]
        y = e[i] * u[:, i]
        if not np.allclose(x, y, rtol=1e-05, atol=1e-08):
            print("Error!")

    return e, u


def preprocess_s3d_input(args):
    feat_path = os.path.join(args.path, "feats", "s3d")
    for filename in tqdm(os.listdir(feat_path)):
        feat = np.load(os.path.join(feat_path, filename)) # n_clips x d
        feat = feat.mean(axis=0) # d
        out_path = os.path.join(args.path, "preprocessed", "s3d")
        np.save(os.path.join(out_path, filename), feat)


def preprocess_frozen_input(args):
    feat_path = os.path.join(args.path, "feats", "frozen")
    for filename in tqdm(os.listdir(feat_path)):
        feat = np.load(os.path.join(feat_path, filename)) # n_clips x n_patches x d
        feat = feat[:,0,:].mean(axis=0) # n_clips x d -> d
        out_path = os.path.join(args.path, "preprocessed", "frozen")
        np.save(os.path.join(out_path, filename), feat)


def preprocess_frozen_graph_input(args):
    feat_path = os.path.join(args.path, "feats", "frozen")
    graph_path = os.path.join(args.path, "graphs")
    out_path = os.path.join(args.path, "preprocessed", "frozen+graph")
    moma = MOMA(args.path)
    
    for split in ["train", "val", "test"]:
        ids_act = moma.get_ids_act(split=split)
        for act in tqdm(moma.get_anns_act(ids_act=ids_act)):
            if act.id == "1YzGUyM3P2k": # issue: no graph
                continue
            if os.path.exists(os.path.join(out_path, f"{act.id}.pt")):
                continue

            nidx, t = 0, 0 
            node2idx, obj_list = {}, []
            for sact in moma.get_anns_sact(ids_sact=act.ids_sact):
                for hoi in moma.get_anns_hoi(ids_hoi=sact.ids_hoi):
                    img = cv2.imread(os.path.join(args.path, "videos", "interaction", f"{hoi.id}.jpg"))
                    img = img[...,::-1]
                    h, w = img.shape[0], img.shape[1]
                    h_s, w_s = 224. / h, 224. / w

                    for entity in (hoi.actors + hoi.objects):
                        if entity.id not in node2idx:
                            node2idx[entity.id] = nidx 
                            nidx += 1

                        bbox = entity.bbox
                        min_x, min_y = bbox.x, bbox.y
                        max_x, max_y = min_x + bbox.width, min_y + bbox.height
                        min_x, min_y = int(min_x * w_s), int(min_y * h_s)
                        max_x, max_y = int(max_x * w_s), int(max_y * h_s)

                        obj_list.append(
                            {
                                "hid": hoi.id,
                                "eid": entity.id,
                                "t": t,
                                "nidx": node2idx[entity.id],
                                "bbox": (min_x, min_y, max_x, max_y),
                            }
                        )

                    t += 1

            n, id2eig = nidx, {}
            for sact in moma.get_anns_sact(ids_sact=act.ids_sact):
                for hoi in moma.get_anns_hoi(ids_hoi=sact.ids_hoi):
                    adj = np.zeros((n, n))
                    for predicate in (hoi.ias + hoi.tas + hoi.atts + hoi.rels):
                        id_src = node2idx[predicate.id_src]
                        if predicate.id_trg is None:
                            id_trg = node2idx[predicate.id_src]
                        else:
                            id_trg = node2idx[predicate.id_trg]    
                        adj[id_src][id_trg] = 1.
                        adj[id_trg][id_src] = 1.
                    
                    e, u = get_laplacian_eig(adj) # n, n x n
                    idx = e.argsort()
                    e, u = e[idx], np.real(u[:,idx])

                    id2eig[hoi.id] = {"e": e, "u": u}
                    torch.save(id2eig, os.path.join(graph_path, f"{hoi.id}.pt"))

            feat = np.load(os.path.join(feat_path, f"{act.id}.npy")) # n_clips x n_patches x d
            x, pos_enc, ids_hoi, ids_entity, ts_l = get_object_repr(feat, obj_list, id2eig)

            data = {
                "x": x,
                "cls": feat[:,0,:].mean(axis=0),
                "pos_enc": pos_enc,
                "ids_hoi": ids_hoi,
                "ids_entity": ids_entity,
                "ts_l": ts_l,
            }
            torch.save(data, os.path.join(out_path, f"{act.id}.pt"))


def preprocess_anno(args):
    moma = MOMA(dir_moma=args.path, paradigm=args.paradigm)
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

    id2cembs = {}
    id2seq, sid2cname = {}, {}
    
    for split in ["train", "val"]:
        ids_act = moma.get_ids_act(split=split)
        anns_act = moma.get_anns_act(ids_act=ids_act)
        
        anno = []
        for act in tqdm(anns_act, desc=f"MOMA-LRG preprocessing ({split})"):
            if act.id == "1YzGUyM3P2k": # issue: no graph
                continue 

            payload = {
                "video_id": act.id,         # video id e.g. '-49z-lj8eYQ'
                "activity_id": act.cid,     # activity class id e.g. 2
                "activity_name": act.cname, # activity name e.g. "basketball game"
            }
            anno.append(payload)
            
            if args.caption == "moma-base":
                captions = []
                for sact in moma.get_anns_sact(act.ids_sact):
                    if captions and sact.cname == captions[-1]:
                        continue
                    captions.append(sact.cname)

                with torch.no_grad():
                    embeddings = sbert_model.encode(captions)
                    id2cembs[act.id] = torch.from_numpy(embeddings)
            elif args.caption == "moma-dtw":
                sact_seq = []
                for sact in moma.get_anns_sact(ids_sact=act.ids_sact):
                    sid2cname[sact.cid] = sact.cname
                    sact_seq.append(sact.cid)
                id2seq[act.id] = np.array(sact_seq)
            elif args.caption == "tsformer":
                caption_path = os.path.join(args.path, "timesformer_captions")
                with open(os.path.join(caption_path, f"{act.id}.txt")) as f:
                    captions = f.readlines()
                captions = [caption.strip() for caption in captions]

                with torch.no_grad():
                    embeddings = sbert_model.encode(captions)
                    id2cembs[act.id] = torch.from_numpy(embeddings)
            else:
                raise NotImplementedError
                
    #     with open(f"anno/moma/{args.paradigm}_{split}.ndjson", "w") as f:
    #         ndjson.dump(anno, f)
                
    # torch.save(id2cembs, f"anno/moma/{args.paradigm}_id2cembs.pt")

    if args.caption == "moma-dtw":
        sid2cemb = {}
        for vid, cname in sid2cname.items():
            cemb = sbert_model.encode(cname)
            cemb = torch.from_numpy(cemb).float()
            sid2cemb[vid] = cemb

        cembs = torch.zeros(len(sid2cemb), cemb.shape[-1])
        for idx, emb in sid2cemb.items():
            cembs[idx] = emb

        cembs = F.normalize(cembs, dim=-1) 
        sact_cemb_sim = torch.mm(cembs, cembs.t()) # 91 x 91
        sact_cemb_sim = sact_cemb_sim.numpy() 

        torch.save(id2seq, "anno/moma/id2seq.pt")
        torch.save(sact_cemb_sim, "anno/moma/sact_cemb_sim.pt")
                
    
def main():
    parser = argparse.ArgumentParser("Interface of preprocessing MOMA-LRG for video retrieval")
    
    parser.add_argument("--path", default="/data/dir_moma")
    parser.add_argument("--paradigm", default="standard")
    parser.add_argument("--model", default="frozen+graph")
    parser.add_argument("--caption", default="moma")
    
    args = parser.parse_args()
    
    preprocess_anno(args)

    # if args.model == "s3d":
    #     preprocess_s3d_input(args)
    # elif args.model == "frozen":
    #     preprocess_frozen_input(args)
    # elif args.model == "frozen+graph":
    #     preprocess_frozen_graph_input(args)
    # else:
    #     raise NotImplementedError
    
    
if __name__ == "__main__":
    main()