import pandas as pd
import root.scripts.LEX as LEX


def main():
    data = pd.read_csv("root/datasets/FakeNewsFilipino.csv")


    try:
        lexFeatures = pd.read_csv("root/datasets/LexFeatures.csv")
    except FileNotFoundError:
        columns = "ttr,root_ttr,corr_ttr,log_ttr,noun_tr,verb_tr,lexical_density,foreign_tr,compound_tr"
        with open("root/datasets/LexFeatures.csv", 'w') as f:
            f.write(columns)
        lexFeatures = pd.read_csv("root/datasets/LexFeatures.csv")

    list_of_vals = []
    counter = 0

    ## Check last progress of LexFeatures.csv and changed index as needed
    for i in data.itertuples():
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
        
        if ((counter+1)%5) == 0:
            lexFeatures = pd.concat([lexFeatures, pd.DataFrame(list_of_vals, index=list(range(len(list_of_vals))))])
            lexFeatures.to_csv("root/datasets/LexFeatures.csv", index=False)
            list_of_vals = []
        
        counter += 1

if __name__ == "__main__":
    main()