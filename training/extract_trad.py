import pandas as pd
import root.scripts.TRAD as TRAD

def extract_trad():
    data = pd.read_csv("root/datasets/FakeNewsPhilippines2024_Lupac.csv")

    try:
        trad_features = pd.read_csv("root/datasets/TradFeatures.csv")
    except FileNotFoundError:
        columns = "word_count,sentence_count,polysyll_count,ave_word_length,ave_phrase_count,ave_syllable_count_of_word,word_count_per_sentence"
        with open("root/datasets/TradFeatures.csv", 'w') as f:
            f.write(columns)
        trad_features = pd.read_csv("root/datasets/TradFeatures.csv")

    list_of_vals = [ ]
    starting_index = -1  # TODO: Check last progress of the corresponding .csv and change starting index as needed

    # word_count_per_doc = TRAD.word_count_per_doc(data['article'][0])
    # sentence_count_per_doc = TRAD.sentence_count_per_doc(data['article'][0])
    # polysyll_count_per_doc = TRAD.polysyll_count_per_doc(data['article'][0])
    # ave_word_length = TRAD.ave_word_length(data['article'][0])
    # ave_phrase_count_per_doc = TRAD.ave_phrase_count_per_doc(data['article'][0])
    # ave_syllable_count_of_word = TRAD.ave_syllable_count_of_word(data['article'][0])
    # word_count_per_sentence = TRAD.word_count_per_sentence(data['article'][0])
    
    # list_of_vals.append({
    #             "word_count": word_count_per_doc,
    #             "sentence_count": sentence_count_per_doc,
    #             "polysyll_count": polysyll_count_per_doc,
    #             "ave_word_length": ave_word_length,
    #             "ave_phrase_count": ave_phrase_count_per_doc,
    #             "ave_syllable_count_of_word": ave_syllable_count_of_word,
    #             "word_count_per_sentence": word_count_per_sentence
    #         })
    # trad_features = pd.concat([pd.DataFrame(list_of_vals, index=list(range(len(list_of_vals)))), trad_features])
    # trad_features.to_csv("root/datasets/TradFeatures.csv", index=False)
    # exit(0)
    
    for i in data.itertuples():
        if (i.Index > starting_index):
            word_count_per_doc = TRAD.word_count_per_doc(i.article)
            sentence_count_per_doc = TRAD.sentence_count_per_doc(i.article)
            polysyll_count_per_doc = TRAD.polysyll_count_per_doc(i.article)
            ave_word_length = TRAD.ave_word_length(i.article)
            ave_phrase_count_per_doc = TRAD.ave_phrase_count_per_doc(i.article)
            ave_syllable_count_of_word = TRAD.ave_syllable_count_of_word(i.article)
            word_count_per_sentence = TRAD.word_count_per_sentence(i.article)

            list_of_vals.append({
                "word_count": word_count_per_doc,
                "sentence_count": sentence_count_per_doc,
                "polysyll_count": polysyll_count_per_doc,
                "ave_word_length": ave_word_length,
                "ave_phrase_count": ave_phrase_count_per_doc,
                "ave_syllable_count_of_word": ave_syllable_count_of_word,
                "word_count_per_sentence": word_count_per_sentence
            })
            
            print(i.Index)
            
            if ((i.Index)%25) == 0:
                print("Saving figures...")
                trad_features = pd.concat([trad_features, pd.DataFrame(list_of_vals, index=list(range(len(list_of_vals))))])
                trad_features.to_csv("root/datasets/TradFeatures.csv", index=False)
                list_of_vals = []
                
    trad_features = pd.concat([trad_features, pd.DataFrame(list_of_vals, index=list(range(len(list_of_vals))))])
    trad_features.to_csv("root/datasets/TradFeatures.csv", index=False)

def main():
    extract_trad()
    
if __name__ == "__main__":
    main()