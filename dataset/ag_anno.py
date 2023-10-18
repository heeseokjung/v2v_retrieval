import os
import csv
import ndjson


def main():
    ##################### Action Genome ########################
    graph_vid = os.listdir("/data/action_genome/graphs")
    graph_vid = [x[:-3] for x in graph_vid]
    ############################################################
    
    ############################# Charades-STA #############################
    train = {}
    with open("/data/action_genome/annotations/charades_sta_train.txt", "r") as f:
        lines = f.readlines()
        for row in lines:
            vid = row.split(" ")[0]
            caption = row.split("##")[-1].rstrip()
            
            if vid not in graph_vid:
                continue
            
            if vid in train:
                train[vid].append(caption)
            else:
                train[vid] = [caption]
                
    train_anno = []
    for vid, caption_list in train.items():
        train_anno.append(
            {
                "vid": vid,
                "captions": ";".join(caption_list),
            }
        )
                
    test = {}
    with open("/data/action_genome/annotations/charades_sta_test.txt", "r") as f:
        lines = f.readlines()
        for row in lines:
            vid = row.split(" ")[0]
            caption = row.split("##")[-1].rstrip()
            
            if vid not in graph_vid:
                continue
            
            if vid in test:
                test[vid].append(caption)
            else:
                test[vid] = [caption]
                
    test_anno = []
    for vid, caption_list in test.items():
        test_anno.append(
            {
                "vid": vid,
                "captions": ";".join(caption_list),
            }
        )
    ############################# Charades-STA #############################
        
    with open("/data/action_genome/annotations/ag_anno_train.ndjson", "w") as f:
        ndjson.dump(train_anno, f)
    with open("/data/action_genome/annotations/ag_anno_test.ndjson", "w") as f:
        ndjson.dump(test_anno, f)
    
    print(len(train_anno))
    print(len(test_anno))
                
    
if __name__ == "__main__":
    main()