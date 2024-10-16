import pandas as pd
import root.scripts.MORPH as MORPH

def extract_morph():
    data = pd.read_csv("root/datasets/FakeNewsPhilippines2024_Lupac.csv")
    
    try:
        morph_features = pd.read_csv("root/datasets/MorphFeatures.csv")
    except FileNotFoundError:
        columns = ("prefix_token_ratio,prefix_derived_ratio,suffix_token_ratio,suffix_derived_ratio,total_affix_token_ratio,"
                   "total_affix_derived_ratio,actor_focus_ratio,object_focus_ratio,benefactive_focus_ratio,locative_focus_ratio,"
                   "instrumental_focus_ratio,referential_focus_ratio,infinitive_verb_ratio,participle_verb_ratio,"
                   "perfective_verb_ratio,imperfective_verb_ratio,contemplative_verb_ratio,recent_past_verb_ratio,aux_verb_ratio")
        with open("root/datasets/MorphFeatures.csv", 'w') as f:
            f.write(columns)
        morph_features = pd.read_csv("root/datasets/MorphFeatures.csv")

    list_of_vals = [ ]
    starting_index = -1  # TODO: Check last progress of the corresponding .csv and change starting index as needed
    
    for i in data.itertuples():
        if (i.Index > starting_index):
            (
                prefix_token_ratio, prefix_derived_ratio, suffix_token_ratio,
                suffix_derived_ratio, total_affix_token_ratio, total_affix_derived_ratio
            ) = MORPH.get_derivational_morph(i.article)
            (
                actor_focus_ratio, object_focus_ratio, benefactive_focus_ratio,
                locative_focus_ratio, instrumental_focus_ratio, referential_focus_ratio,
                infinitive_verb_ratio, participle_verb_ratio, perfective_verb_ratio,
                imperfective_verb_ratio, contemplative_verb_ratio, recent_past_verb_ratio,
                aux_verb_ratio
            ) = MORPH.get_inflectional_morph(i.article)

            list_of_vals.append({
                "prefix_token_ratio" :prefix_token_ratio,
                "prefix_derived_ratio" : prefix_derived_ratio,
                "suffix_token_ratio" : suffix_token_ratio,
                "suffix_derived_ratio" : suffix_derived_ratio,
                "total_affix_token_ratio" : total_affix_token_ratio,
                "total_affix_derived_ratio" : total_affix_derived_ratio,
                "actor_focus_ratio" : actor_focus_ratio,
                "object_focus_ratio" : object_focus_ratio,
                "benefactive_focus_ratio" : benefactive_focus_ratio,
                "locative_focus_ratio" : locative_focus_ratio,
                "instrumental_focus_ratio" : instrumental_focus_ratio,
                "referential_focus_ratio" : referential_focus_ratio,
                "infinitive_verb_ratio" : infinitive_verb_ratio,
                "participle_verb_ratio" : participle_verb_ratio,
                "perfective_verb_ratio" : perfective_verb_ratio,
                "imperfective_verb_ratio" : imperfective_verb_ratio,
                "contemplative_verb_ratio" : contemplative_verb_ratio,
                "recent_past_verb_ratio" : recent_past_verb_ratio,
                "aux_verb_ratio" : aux_verb_ratio
                })
            
            print(i.Index)
            
            if ((i.Index)%25) == 0:
                print("Saving features...")
                morph_features = pd.concat([morph_features, pd.DataFrame(list_of_vals, index=list(range(len(list_of_vals))))])
                morph_features.to_csv("root/datasets/MorphFeatures.csv", index=False)
                list_of_vals = []

    #   Write remaining values to csv
    morph_features = pd.concat([morph_features, pd.DataFrame(list_of_vals, index=list(range(len(list_of_vals))))])
    morph_features.to_csv("root/datasets/MorphFeatures.csv", index=False)

def main():
    extract_morph()
    
if __name__ == "__main__":
    main()