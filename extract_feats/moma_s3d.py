import os
import math
import numpy as np
import torch
import ffmpeg

from momaapi import MOMA
from s3dg import S3D
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class MOMARawVideoDataset(Dataset):
    def __init__(self, video_ids):
        self.in_path = "/data/dir_moma/videos/raw"
        self.video_ids = video_ids
        self.fps = 16

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
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
        video = torch.from_numpy(video.astype("float32"))
        video = video.permute(0, 3, 1, 2) # n_frames x 3 x h x w
        video = video / 255.

        # zero padding
        if video.shape[0] % self.fps:
            d = self.fps - (video.shape[0] % self.fps)
            z = torch.zeros(d, 3, 224,224)
            video = torch.cat((video, z), dim=0)

        video = video.view(-1, self.fps, 3, 224, 224) # sec x fps x 3 x 224 x 224
        video = video.transpose(1, 2) # sec x 3 x fps x 224 x 224

        return {
            "vid": vid,
            "video": video,
        }


def main():
    model = S3D("s3d_dict.npy", 512)
    model.load_state_dict(torch.load("s3d_howto100m.pth"))
    model = model.eval()
    model = model.to("cuda") 

    moma = MOMA("/data/dir_moma")
    ids_act = moma.get_ids_act()

    dataset = MOMARawVideoDataset(video_ids=ids_act)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    bs = 16 # batch size
    out_path = "/data/dir_moma/feats/s3d"
    for batch in tqdm(dataloader):
        vid = batch["vid"][0]
        video = batch["video"].squeeze(dim=0) # sec x 3 x fps x 224 x 224

        features = torch.zeros((video.shape[0], 1024))
        n_iter = int(math.ceil(video.shape[0] / bs))
        for i in range(n_iter):
            s = i * bs
            e = (i + 1) * bs
            x = video[s:e, ...].to("cuda")
            with torch.no_grad():
                feat = model(x)["mixed_5c"]
            features[s:e,:] = feat.detach().cpu()

        features = features.numpy()
        np.save(os.path.join(out_path, f"{vid}.npy"), features)


if __name__ == "__main__":
    main()