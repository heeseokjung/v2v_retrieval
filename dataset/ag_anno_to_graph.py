import os
import pickle

import torch
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from sentence_transformers import SentenceTransformer


embedding_cache = {}


def sbert_embedding(model, sentence):
    '''
        Args:
            - model: Sentence BERT pre-trained model
            - sentence: string
        Do:
            - obtain sentence embedding for given sentence
    '''
    
    global embedding_cache
    
    if sentence in embedding_cache:
        return embedding_cache[sentence]
    
    embedding = model.encode(sentence)
    embedding = torch.tensor(embedding) # d: 384
    embedding_cache[sentence] = embedding
    
    return embedding


def convert_anno_to_graph(anno, model, ag_path):
    vid_2_graphs = {}
    for idx, objects in tqdm(anno.items()):
        vid = idx.split("/")[0][:-4]
        fid = idx.split("/")[-1]
        
        data = Data()
        
        entities = ["person"]
        edge_index, edge_attr = [], []
        for obj in objects:
            entities.append(obj["class"])
            obj_idx = len(entities) - 1
            
            rels = []
            if obj["attention_relationship"] is not None:
                rels = rels + obj["attention_relationship"]
            if obj["spatial_relationship"] is not None:
                rels = rels + obj["spatial_relationship"]
            if obj["contacting_relationship"] is not None:
                rels = rels + obj["contacting_relationship"]
                
            for rel in rels:
                rel = rel.replace("_", " ")
                rel_emb = sbert_embedding(model, rel)
                edge_index.append([0, obj_idx])
                edge_attr.append(rel_emb)
                
        if entities:
            entities = [sbert_embedding(model, x) for x in entities]
            data.x = torch.stack(entities, dim=0)
        if edge_index:
            data.edge_index = torch.LongTensor(edge_index).t().contiguous()
        if edge_attr:
            data.edge_attr = torch.stack(edge_attr, dim=0)
        
        if edge_index:
            if vid in vid_2_graphs:
                vid_2_graphs[vid].append(data)
            else:
                vid_2_graphs[vid] = [data]
            
    os.makedirs(os.path.join(ag_path, "graphs"), exist_ok=True)
    for vid, data_list in tqdm(vid_2_graphs.items()):
        batch = Batch.from_data_list(data_list)
        torch.save(
            {"vid": vid, "data": batch},
            os.path.join(ag_path, "graphs", f"{vid}.pt")
        )
        

def main():
    # Action Genome dataset path
    ag_path = "/data/action_genome/"
    
    # Sentence BERT pre-trained model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model = model.to("cuda")
        
    with open(
        os.path.join(ag_path, "annotations", "object_bbox_and_relationship.pkl"), "rb"
    ) as f:
        anno = pickle.load(f)
        
    convert_anno_to_graph(anno, model, ag_path)
    
    
if __name__ == "__main__":
    main()