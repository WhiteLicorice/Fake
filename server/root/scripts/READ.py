from .TRAD import word_count_per_doc, word_count_per_sentence

def compute_readability_score(text):
    fxn_const = -4.161      ##  Function constant
    wps_const = 0.280       ##  Word count per sentence weight
    cw_const = 0.106        ##  Word count weight
    
    #print(text)
    
    ##  Get total word count and average word count per sentence
    wps = word_count_per_sentence(text)
    cw = word_count_per_doc(text)
    
    readability_score = wps_const*(wps) + cw_const*(cw) + fxn_const
    
    ##  Clamp readability score to non-negative values since:
    """
        4.9 and below       Very Easy
        5-6                 Easy
        7-8                 Average
        9-10                Difficult
        11 and up           Very Difficult
    """
    if readability_score < 0:
        readability_score = 0
    
    return readability_score