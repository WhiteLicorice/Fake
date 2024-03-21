import pandas as pd
import root.scripts.OOV as OOV

def extract_oov():
    data = pd.read_csv("root/datasets/FakeNewsPhilippines2024_Lupac.csv")

    try:
        oov_features = pd.read_csv("root/datasets/OovFeatures.csv")
    except FileNotFoundError:
        columns = "count_oov_words"
        with open("root/datasets/OovFeatures.csv", 'w') as f:
            f.write(columns)
        oov_features = pd.read_csv("root/datasets/OovFeatures.csv")

    list_of_vals = [ ]
    starting_index = -1  # TODO: Check last progress of the corresponding .csv and change starting index as needed
    
    # count_oov_words = OOV.count_oov_words(data['article'][0])
    # list_of_vals.append({
    # "count_oov_words": count_oov_words,
    # })

    # oov_features = pd.concat([pd.DataFrame(list_of_vals, index=list(range(len(list_of_vals)))), oov_features])
    # oov_features.to_csv("root/datasets/OovFeatures.csv", index=False)

    # exit(0)

    for i in data.itertuples():
        if (i.Index > starting_index):
            count_oov_words = OOV.count_oov_words(i.article)
            
            #   Append to list
            list_of_vals.append({
                "count_oov_words": count_oov_words,
            })
            
            print(i.Index)
            
            if ((i.Index)%25) == 0:
                print("Saving figures...")
                oov_features = pd.concat([oov_features, pd.DataFrame(list_of_vals, index=list(range(len(list_of_vals))))])
                oov_features.to_csv("root/datasets/OovFeatures.csv", index=False)
                list_of_vals = []
                
    oov_features = pd.concat([oov_features, pd.DataFrame(list_of_vals, index=list(range(len(list_of_vals))))])
    oov_features.to_csv("root/datasets/OovFeatures.csv", index=False)

def main():
    extract_oov()
    
if __name__ == "__main__":
    main()