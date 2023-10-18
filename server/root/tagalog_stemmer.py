import copy
VOWELS = 'aeiou'
PREFIXES = [
    'ma', 'na', 'pa', 'mag', 'pag', 'nag', 'i'
]
INFIX = [
    'um', 'in',
]
SUFFIX = [
    'han', 'hin', 
    'ing', 'ang', 
    'ng', 'an', 
    'in',
]
def sort_list_by_str_len(lst):
    return sorted(copy.deepcopy(lst), key=lambda p: len(p), reverse=True)
def prefix_remover(data):
    stemmed = data
    for prefix in sort_list_by_str_len(PREFIXES):
       p_len = len(prefix)
       if prefix == stemmed[0:p_len]:
            stemmed = stemmed[p_len:len(stemmed)]
            break
    first_half = stemmed[0:2]
    future_stemmed = stemmed[2:len(stemmed)]
    if first_half * 2 == stemmed[0:4] and first_half != future_stemmed:
        stemmed = future_stemmed
    if len(stemmed) >= 2 and stemmed[0] in VOWELS and stemmed[0] == stemmed[1]:
        stemmed = stemmed[1:len(stemmed)]
    return stemmed
def infix_remover(data):
    stemmed = data
    for infix in INFIX:
        shift = 1
        infix_end_pos = len(infix)+shift
        if stemmed[shift:len(infix)+shift] == infix:
            stemmed = '{}{}'.format(stemmed[0],stemmed[infix_end_pos:len(stemmed)])
            stemmed = prefix_remover(stemmed)
            break
    return stemmed
def suffix_remover(data):
    stemmed = data
    for suffix in sort_list_by_str_len(SUFFIX):
        stemmed_len = len(stemmed)
        end_suffix_start = stemmed_len-len(suffix)
        if stemmed[end_suffix_start:stemmed_len] == suffix:
            stemmed = stemmed[0:end_suffix_start]
            break
    return stemmed
def stemmer(data):
    stemmed = data.lower()
    stemmed = prefix_remover(stemmed)
    stemmed = infix_remover(stemmed)
    return  suffix_remover(stemmed)
if __name__ == "__main__":
    stemmer('ginagandahan')
