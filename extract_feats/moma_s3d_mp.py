import os
import numpy as np
import torch
import ffmpeg
import multiprocessing as mp

from momaapi import MOMA
from s3dg import S3D
from tqdm import tqdm


def load_video(path, vid, fps=16):
    cmd = (
        ffmpeg
            .input(os.path.join(path, f"{vid}.mp4"))
            .filter("fps", fps=fps)
            .filter("scale", 224, 224)
    )
    out, _ = (
        cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run(capture_stdout=True, quiet=True)
    )

    video = np.frombuffer(out, np.uint8).reshape([-1, 224, 224, 3])
    video = torch.from_numpy(video.astype("float32")) # n_frames x h x w x 3
    n_frames = video.shape[0]
    duration = int(n_frames / fps) # in second
    video = video[:duration * fps, ...] # duration*fps x h x w x 3
    video = video.reshape(duration, fps, 224, 224, 3)
    video = video.permute(0, 4, 1, 2, 3) # duration x 3 x fps x h x w
    video = video / 255.

    return video


def worker(video_ids, rank):
    in_path = "/data/dir_moma/videos/raw"
    out_path = "/data/dir_moma/feats/s3d"

    # Load S3D model pre-trained on HowTo100M using MIL-NCE
    model = S3D("s3d_dict.npy", 512)
    model.load_state_dict(torch.load("s3d_howto100m.pth"))
    model = model.eval()
    model = model.to(f"cuda:{rank}") 

    for vid in tqdm(video_ids, desc=f"RANK [{rank}]"):
        video = load_video(in_path, vid, fps=16)
        video = video.to(f"cuda:{rank}")
        feat_list = []
        for x in video:
            x = x.unsqueeze(dim=0) # 1 x 3 x 16 x 224 x 224
            with torch.no_grad():
                feat = model(x)["mixed_5c"] # 1 x 1024
            feat = feat.detach().cpu()
            feat_list.append(feat)
        
        s3d_emb = torch.cat(feat_list, dim=0).numpy()
        np.save(os.path.join(out_path, f"{vid}.npy"), s3d_emb)


def main():
    n_gpus = 3
    moma = MOMA("/data/dir_moma")
    ids_act = moma.get_ids_act()
    chunk_size = int(len(ids_act) / n_gpus)

    torch.multiprocessing.set_start_method('spawn')
    
    procs = []
    for i in range(n_gpus):
        if i < n_gpus - 1:
            p = mp.Process(target=worker, args=(ids_act[i*chunk_size:(i+1)*chunk_size], i))
        else:
            p = mp.Process(target=worker, args=(ids_act[i*chunk_size:], i))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()