import decord
import random
import numpy as np


def s3d_clip_sampler(path, clip_duration, frames_per_clip, stride):
    video_reader = decord.VideoReader(path)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / fps
    n_clips = max(int(duration / clip_duration), 1)
    
    video = []
    clip_split = np.linspace(start=0, stop=vlen, num=n_clips+1).astype(int)
    for i in range(n_clips):
        s = clip_split[i]
        e = clip_split[i+1] - frames_per_clip
        j = random.choice(range(s, e))
        idx = np.arange(j, j + frames_per_clip, stride)
        clip = video_reader.get_batch(idx).asnumpy() # n_frames x h x w x 3
        clip = [clip[i] for i in range(len(clip))]
        video.append(clip)     

    return video


def frozen_clip_sampler(path, clip_duration, num_frames):
    video_reader = decord.VideoReader(path)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / fps
    n_clips = max(int(duration / clip_duration), 1)

    video = []
    clip_split = np.linspace(start=0, stop=vlen, num=n_clips+1).astype(int)
    for i in range(n_clips):
        idx = []
        cs = clip_split[i]
        ce = clip_split[i+1]
        sample_ranges = np.linspace(start=cs, stop=ce, num=num_frames+1).astype(int)
        for j in range(num_frames):
            idx.append(random.choice(range(sample_ranges[j], sample_ranges[j+1])))
        clip = video_reader.get_batch(np.array(idx)).asnumpy() # n_frames x h x w x 3
        video.append(clip)

    video = np.stack(video, axis=0) # n_clips x n_frames x h x w x 3
    
    return video