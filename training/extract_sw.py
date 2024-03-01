import pandas as pd
import root.scripts.SW as SW

def extract_sw():
    data = pd.read_csv("root/datasets/FakeNewsFilipino.csv")

    try:
        sw_features = pd.read_csv("root/datasets/SwFeatures.csv")
    except FileNotFoundError:
        columns = "count_stopwords"
        with open("root/datasets/SwFeatures.csv", 'w') as f:
            f.write(columns)
        sw_features = pd.read_csv("root/datasets/SwFeatures.csv")

    list_of_vals = [ ]
    starting_index = -1  # TODO: Check last progress of the corresponding .csv and change starting index as needed
    
    count_stopwords = SW.count_stopwords(data['article'][0])
    list_of_vals.append({
        "count_stopwords": count_stopwords,
    })
    sw_features = pd.concat([pd.DataFrame(list_of_vals, index=list(range(len(list_of_vals)))), sw_features])
    sw_features.to_csv("root/datasets/SwFeatures.csv", index=False)
    exit(0)

    for i in data.itertuples():
        if (i.Index > starting_index):
            count_stopwords = SW.count_stopwords(i.article)
            
            #   Append to list
            list_of_vals.append({
                "count_stopwords": count_stopwords,
            })
            
            print(i.Index)
            
            if ((i.Index)%25) == 0:
                print("Saving figures...")
                sw_features = pd.concat([sw_features, pd.DataFrame(list_of_vals, index=list(range(len(list_of_vals))))])
                sw_features.to_csv("root/datasets/SwFeatures.csv", index=False)
                list_of_vals = []
                
    sw_features = pd.concat([sw_features, pd.DataFrame(list_of_vals, index=list(range(len(list_of_vals))))])
    sw_features.to_csv("root/datasets/SwFeatures.csv", index=False)

def main():
    extract_sw()
    
if __name__ == "__main__":
    main()