import os
import ndjson
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer


class ActionGenomeRetrievalTrainDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        
        
class ActionGenomeRetrievalEvalDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        
        self.path = path
        with open(
            os.path.join(path, "annotations", "ag_anno_test.ndjson"), "r"
        ) as f:
            self.anno = ndjson.load(f)
            
        self._get_caption_embeddings()
        self._prepare_batches()
        
    def _get_caption_embeddings(self):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        model = model.to("cuda")
        
        self.embedding_lookup = {}
        for video in self.anno:
            vid = video["vid"]
            captions = video["captions"].split(";")
            
            with torch.no_grad():
                caption_embedding = model.encode(captions)
            
            caption_embedding = torch.tensor(caption_embedding) # n_caption x d
            self.embedding_lookup[vid] = caption_embedding
            
    def _prepare_batches(self):
        self.batches, self.graph_cache = {}, {}
        for video_x in self.anno:
            vids, captions, similarity = [], [], []
            for video_y in self.anno:
                if video_x["vid"] == video_y["vid"]:
                    continue
                
                vids.append(video_y["vid"])
                captions.append(video_y["captions"].split(";"))
                similarity.append(
                    self.get_proxy_similarity(video_x["vid"], video_y["vid"])
                )
                
            self.batches[video_x["vid"]] = {
                "query_vid": video_x["vid"],
                "query_captions": video_x["captions"].split(";"),
                "vids": vids,
                "similarity": torch.tensor(similarity),
            }
            
            self.graph_cache[video_x["vid"]] = self.load_graph(video_x["vid"])
                
    def get_proxy_similarity(self, vid1, vid2):
        emb_1 = self.embedding_lookup[vid1].to("cuda")
        emb_1 = F.normalize(emb_1, dim=-1) # n_captions x d
        
        emb_2 = self.embedding_lookup[vid2].to("cuda")
        emb_2 = F.normalize(emb_2, dim=-1) # n_captions x d
        
        return torch.mm(emb_1, emb_2.t()).mean()
    
    def load_graph(self, vid):
        graph_path = os.path.join(self.path, "graphs", f"{vid}.pt")
        graphs = torch.load(graph_path)
        
        return graphs
    
    def __len__(self):
        return len(self.anno)
        
    def __getitem__(self, idx):
        batch = self.batches[self.anno[idx]["vid"]]
        batch["query_graphs"] = self.graph_cache[batch["query_vid"]]
        batch["graphs"] = [self.graph_cache[vid] for vid in batch["vids"]]
        
        return batch