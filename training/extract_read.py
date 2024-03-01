import pandas as pd
import root.scripts.READ as READ

def extract_read():
    data = pd.read_csv("root/datasets/FakeNewsFilipino.csv")

    try:
        read_features = pd.read_csv("root/datasets/ReadFeatures.csv")
    except FileNotFoundError:
        columns = "readability_score"
        with open("root/datasets/ReadFeatures.csv", 'w') as f:
            f.write(columns)
        read_features = pd.read_csv("root/datasets/ReadFeatures.csv")

    list_of_vals = [ ]
    starting_index = -1  # TODO: Check last progress of the corresponding .csv and change starting index as needed
    
    # readability_score = READ.compute_readability_score(data['article'][0])
    # list_of_vals.append({
    #     "readability_score": readability_score,
    # })

    # read_features = pd.concat([pd.DataFrame(list_of_vals, index=list(range(len(list_of_vals)))), read_features])
    # read_features.to_csv("root/datasets/ReadFeatures.csv", index=False)

    # exit(0)
    for i in data.itertuples():
        if (i.Index > starting_index):
            readability_score = READ.compute_readability_score(i.article)
            
            #   Append to list
            list_of_vals.append({
                "readability_score": readability_score,
            })
            
            print(i.Index)
            
            if ((i.Index)%25) == 0:
                print("Saving figures...")
                read_features = pd.concat([read_features, pd.DataFrame(list_of_vals, index=list(range(len(list_of_vals))))])
                read_features.to_csv("root/datasets/ReadFeatures.csv", index=False)
                list_of_vals = []
                
    read_features = pd.concat([read_features, pd.DataFrame(list_of_vals, index=list(range(len(list_of_vals))))])
    read_features.to_csv("root/datasets/ReadFeatures.csv", index=False)

def main():
    extract_read()
    
if __name__ == "__main__":
    main()