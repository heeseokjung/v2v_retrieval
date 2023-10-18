import os
import cv2
import numpy as np
from momaapi import MOMA
from tqdm import tqdm


def main():
    moma = MOMA("/data/dir_moma", paradigm="standard")

    path = "/data/dir_moma/bbox"
    for split in ["train", "val"]:
        ids_act = moma.get_ids_act(split=split)
        for act in tqdm(moma.get_anns_act(ids_act)):
            for sact in moma.get_anns_sact(act.ids_sact):
                for hoi in moma.get_anns_hoi(sact.ids_hoi):
                    img = cv2.imread(f"/data/dir_moma/videos/interaction/{hoi.id}.jpg")
                    img = img[...,::-1]
                    h, w = img.shape[0], img.shape[1]
                    h_s, w_s = 224. / h, 224. / w

                    bbox_list = []
                    for entity in (hoi.actors + hoi.objects):
                        bbox = entity.bbox
                        min_x, min_y = bbox.x, bbox.y
                        max_x, max_y = min_x + bbox.width, min_y + bbox.height
                        min_x, min_y = int(min_x*w_s), int(min_y*h_s)
                        max_x, max_y = int(max_x*w_s), int(max_y*h_s)
                        bbox_list.append((min_x, min_y, max_x, max_y))
                    bbox_list = np.array(bbox_list)
                    np.save(
                        os.path.join(path, f"{hoi.id}.npy"), bbox_list
                    )


if __name__ == "__main__":
    main()