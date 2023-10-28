import pandas as pd
import numpy as np
from momaapi import MOMA


def main():
    df = pd.read_csv("pilot.csv")
    triplet_video_ids = []
    for idx, row in df.iterrows():
        triplet_video_ids.append(row["qvid"])
        triplet_video_ids.append(row["vid1"])
        triplet_video_ids.append(row["vid2"])
    
    triplet_video_ids = list(set(triplet_video_ids))

    triplet_duration = []
    total_duration = []

    moma = MOMA("/data/dir_moma")
    ids_act = moma.get_ids_act()
    for act in moma.get_anns_act(ids_act=ids_act):
        total_duration.append(act.end - act.start)
        if act.id in triplet_video_ids:
            triplet_duration.append(act.end - act.start)

    triplet_duration = np.array(triplet_duration)
    total_duration = np.array(total_duration)

    print(f"[triplet] mu: {triplet_duration.mean()} std: {triplet_duration.std()}")
    print(f"[total] mu: {total_duration.mean()} std: {total_duration.std()}")




if __name__ == "__main__":
    main()