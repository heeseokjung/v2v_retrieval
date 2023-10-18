import os
import torch
import numpy as np

from momaapi import MOMA
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

min_n = 10000000
max_len = -1


class SentenceEmbeddingModel:
    def __init__(self, name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(name)
        self.cache = {}
        
    def get_sentence_embedding(self, sentence):
        if sentence in self.cache:
            return self.cache[sentence] 
        else:
            with torch.no_grad():
                sentence_emb = self.model.encode(sentence)
                sentence_emb = torch.from_numpy(sentence_emb)
            self.cache[sentence] = sentence_emb
            return sentence_emb
        
        
def refine_sentence(sentence):
    if "[src]" in sentence:
        sentence = sentence.replace("[src]", "")
    if "[trg]" in sentence:
        sentence = sentence.replace("[trg]", "")
        
    return sentence.strip()


def convert_anno_to_graph(moma, model, path):
    for split in ["train", "val", "test"]:
        ids_act = moma.get_ids_act(split=split)
        anns_act = moma.get_anns_act(ids_act)
        
        for act in tqdm(anns_act):
            if split == "train" and act.id == "1YzGUyM3P2k": # issue: no graph
                continue 
            
            '''
                node2idx: Entity.id (e.g. "A", "B", "1", "2", ...) to node index
                nidx2txt: node index to corresponing text (e.g. "customer")
                eidx2txt: edge index to corresponding text (e.g. "is talking to")
            '''
            
            nidx, eidx = 0, 1
            node2idx, eidx2txt = {}, {}
            
            anns_sact = moma.get_anns_sact(ids_sact=act.ids_sact)
            
            # preprocessing for node index
            for sact in anns_sact:
                anns_hoi = moma.get_anns_hoi(ids_hoi=sact.ids_hoi)
                for hoi in anns_hoi:
                    for entity in (hoi.actors + hoi.objects):
                        if entity.id not in node2idx:
                            node2idx[entity.id] = nidx
                            nidx += 1

            global min_n
            if nidx < min_n:
                min_n = nidx
            
            # record each interaction
            t = 0
            interactions = {}
            for sact in anns_sact:
                anns_hoi = moma.get_anns_hoi(ids_hoi=sact.ids_hoi)
                for hoi in anns_hoi:
                    # record nodes
                    x = {} # nidx : bbox
                    for entity in (hoi.actors + hoi.objects):
                        bbox = entity.bbox
                        min_x, min_y = bbox.x, bbox.y
                        max_x, max_y = min_x + bbox.width, min_y + bbox.height
                        x[node2idx[entity.id]] = (min_x, min_y, max_x, max_y)

                    # record edges 
                    edge_index = []
                    for predicate in (hoi.ias + hoi.tas + hoi.atts + hoi.rels):
                        id_src = node2idx[predicate.id_src]
                        if predicate.id_trg is None:
                            id_trg = node2idx[predicate.id_src]
                        else:
                            id_trg = node2idx[predicate.id_trg]
                        
                        edge = [id_src, id_trg, eidx]
                        edge_index.append(edge)
                        
                        sentence = refine_sentence(predicate.cname)
                        eidx2txt[eidx] = sentence
                        eidx += 1

                    if edge_index:
                        edge_index = np.array(edge_index, dtype=np.int64).T
                        edge_index = np.ascontiguousarray(edge_index)
                        interactions[hoi.id] = {
                            "time": int(hoi.time - act.start),
                            "x": x,
                            "edge_index": edge_index,
                        }

                    t += 1

            global max_len
            if t > max_len:
                max_len = t

            edge_attr = torch.zeros(len(eidx2txt)+1, 384)
            for idx, txt in eidx2txt.items():
                edge_attr[idx] = model.get_sentence_embedding(txt)

            # data = {"edge_attr": edge_attr, "interactions": interactions}
            # torch.save(data, os.path.join(path, f"{act.id}.pt"))  

    print(f"min_n: {min_n}")
    print(f"max_len: {max_len}")          
    
    
def main():
    path = "/data/dir_moma"
    moma = MOMA(path, paradigm="standard")
    
    model = SentenceEmbeddingModel(name="all-MiniLM-L6-v2")
    
    convert_anno_to_graph(moma, model, os.path.join(path, "graphs"))
    
    
if __name__ == "__main__":
    main()