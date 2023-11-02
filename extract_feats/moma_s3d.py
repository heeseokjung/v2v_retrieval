import os
import numpy as np
import torch
import decord
import ffmpeg

from momaapi import MOMA
from s3dg import S3D
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class MOMALRGRawVideoDataset(Dataset):
    def __init__(self, ids_act):
        self.in_path = "/data/dir_moma/videos/raw"
        self.ids_act = ids_act
        self.fps = 16

    def __len__(self):
        return len(self.ids_act)

    def __getitem__(self, idx):
        vid = self.ids_act[idx]
        cmd = (
            ffmpeg
                .input(os.path.join(self.in_path, f"{vid}.mp4"))
                .filter("fps", fps=self.fps)
                .filter("scale", 224, 224)
        )
        out, _ = (
            cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24")
                .run(capture_stdout=True, quiet=True)
        )

        video = np.frombuffer(out, np.uint8).reshape([-1, 224, 224, 3])
        video = torch.from_numpy(video.astype("float32")) # n_frames x h x w x 3
        n_frames = video.shape[0]
        duration = int(n_frames / self.fps) # in second
        video = video[:duration*self.fps, ...] # duration*fps x h x w x 3
        video = video.reshape(duration, self.fps, 224, 224, 3)
        video = video.permute(0, 4, 1, 2, 3) # duration x 3 x fps x h x w
        video = video / 255.

        return vid, video


def collate_fn(data):
    vid, video = data[0]
    batch = {
        "vid": vid,
        "video": video,
    }
    return batch


def main():
    # Load S3D model pre-trained on HowTo100M using MIL-NCE
    model = S3D("s3d_dict.npy", 512)
    model.load_state_dict(torch.load("s3d_howto100m.pth"))
    model = model.eval()
    model = model.to("cuda") 

    # MOMA-LRG annotation lookup api
    moma = MOMA("/data/dir_moma")
    ids_act = moma.get_ids_act()

    dataset = MOMALRGRawVideoDataset(ids_act)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    for batch in tqdm(dataloader):
        vid = batch["vid"]
        video = batch["video"].to("cuda")
        emb_list = []
        for chunk in video:
            chunck = chunk.unsqueeze(dim=0)
            with torch.no_grad():
                emb = model(chunck)["mixed_5c"] # 1 x 1024
            emb_list.append(emb.detach().cpu())
        feat = torch.cat(emb_list, dim=0)

        print(f"feat: {feat.shape}")
        

        


if __name__ == "__main__":
    main()