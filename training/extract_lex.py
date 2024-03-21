import pandas as pd
import root.scripts.LEX as LEX

def extract_lex():
    data = pd.read_csv("root/datasets/FakeNewsPhilippines2024_Lupac.csv")
    
    try:
        lex_features = pd.read_csv("root/datasets/LexFeatures.csv")
    except FileNotFoundError:
        columns = "ttr,root_ttr,corr_ttr,log_ttr,noun_tr,verb_tr,lexical_density,foreign_tr,compound_tr"
        with open("root/datasets/LexFeatures.csv", 'w') as f:
            f.write(columns)
        lex_features = pd.read_csv("root/datasets/LexFeatures.csv")

    list_of_vals = [ ]
    starting_index = 2876  # TODO: Check last progress of the corresponding .csv and change starting index as needed
    
    for i in data.itertuples():
        if (i.Index > starting_index):
            (ttr, root_ttr, corr_ttr, log_ttr) = LEX.get_type_token_ratios(i.article)
            (noun_tr, verb_tr, lexical_density, foreign_tr, compound_tr) = LEX.get_token_ratios(i.article)

            list_of_vals.append({
                "ttr" : ttr,
                "root_ttr" : root_ttr,
                "corr_ttr" : corr_ttr,
                "log_ttr" : log_ttr,
                "noun_tr" : noun_tr,
                "verb_tr" : verb_tr,
                "lexical_density" : lexical_density,
                "foreign_tr" : foreign_tr,
                "compound_tr" : compound_tr
                })
            print(i.Index)
            
            if ((i.Index)%25) == 0:
                print("Saving figures...")
                lex_features = pd.concat([lex_features, pd.DataFrame(list_of_vals, index=list(range(len(list_of_vals))))])
                lex_features.to_csv("root/datasets/LexFeatures.csv", index=False)
                list_of_vals = []
                
    lex_features = pd.concat([lex_features, pd.DataFrame(list_of_vals, index=list(range(len(list_of_vals))))])
    lex_features.to_csv("root/datasets/LexFeatures.csv", index=False)

def main():
    extract_lex()
    
if __name__ == "__main__":
    main()