import pandas as pd
import root.scripts.MORPH as MORPH


def main():
    data = pd.read_csv("root/datasets/FakeNewsFilipino.csv")
    try:
        morphFeatures = pd.read_csv("root/datasets/MorphFeatures.csv")
    except FileNotFoundError:
        columns = ("prefix_token_ratio,prefix_derived_ratio,suffix_token_ratio,suffix_derived_ratio,total_affix_token_ratio,"
                   "total_affix_derived_ratio,actor_focus_ratio,object_focus_ratio,benefactive_focus_ratio,locative_focus_ratio,"
                   "instrumental_focus_ratio,referential_focus_ratio,infinitive_verb_ratio,participle_verb_ratio,"
                   "perfective_verb_ratio,imperfective_verb_ratio,contemplative_verb_ratio,recent_past_verb_ratio,aux_verb_ratio")
        with open("root/datasets/MorphFeatures.csv", 'w') as f:
            f.write(columns)
        morphFeatures = pd.read_csv("root/datasets/MorphFeatures.csv")

    list_of_vals = []

    ## Check last progress of MorphFeatures.csv and changed index as needed
    for i in data.itertuples():
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
            morphFeatures = pd.concat([morphFeatures, pd.DataFrame(list_of_vals, index=list(range(len(list_of_vals))))])
            morphFeatures.to_csv("root/datasets/MorphFeatures.csv", index=False)
            list_of_vals = []

    # Write remaining values to csv
    morphFeatures = pd.concat([morphFeatures, pd.DataFrame(list_of_vals, index=list(range(len(list_of_vals))))])
    morphFeatures.to_csv("root/datasets/MorphFeatures.csv", index=False)

if __name__ == "__main__":
    main()