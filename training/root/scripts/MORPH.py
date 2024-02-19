from nltk.tag import StanfordPOSTagger
from nltk import word_tokenize, wordpunct_tokenize
import os, math, re
from nltk import data as nltk_data

# DIRECTORIES FROM train.py when calling 'train.py'
if __name__ == "__main__":
	from TRAD import word_count_per_doc, sentence_count_per_doc, cleaner
	nltk_data.path.append("../runtime_env/nltk_data")       # Tell compiler too look for nltk_data at /runtime_env/nltk_data
	java_path = "../runtime_env/custom_java/bin"
	stanford_dir = "../runtime_env/stanford-postagger-full-2020-11-17"

else:
	from root.scripts.TRAD import word_count_per_doc, sentence_count_per_doc, cleaner
	nltk_data.path.append("./root/runtime_env/nltk_data")       # Tell compiler too look for nltk_data at /runtime_env/nltk_data
	java_path = "./root/runtime_env/custom_java/bin"
	stanford_dir = "./root/runtime_env/stanford-postagger-full-2020-11-17"

# MAIN FUNCTIONS

word_count_ = 0
# ----------------------------------------------------------------------------------
# DERIVATIONAL MORPHOLOGY 

def get_derivational_morph(text):
	global word_count
	word_count = word_count_per_doc(text)

	prefix_list = ['ma','pa','na','hing','i','ika','in','ipa','ipag','ipang','ka','ma','mag','magka','magpa','maka','maki','makipag','mang','mapag','may','napaka','pag','pagka','pagkaka','paki','pakikipag','pala','pam','pan','pang','pinaka','tag','taga','tagapag','um']
	suffix_list = ['an','ay','ng','oy','in','ing']

	splitted = re.split('[!?.]+', text)
	splitted = [i for i in splitted if i]

	prefix_count = 0
	suffix_count = 0
	derived_words = []

	# prefix processing
	for i in prefix_list:
		for j in splitted:
			if j.startswith(i):
				prefix_count += 1
				if j not in derived_words:
					derived_words.append(j)

	# suffix processing
	for i in suffix_list:
		for j in splitted:
			if j.endswith(i):
				suffix_count += 1
				if j not in derived_words:
					derived_words.append(j)

	prefix_token_ratio = prefix_count / word_count

	if word_count == 0:
		tuple_of_zeroes = (0,) * 6
		return tuple_of_zeroes
	
	no_of_derived_words = len(derived_words)
	if no_of_derived_words == 0:
		prefix_derived_ratio = 0
		suffix_derived_ratio = 0
		total_affix_derived_ratio = 0

	else:
		prefix_derived_ratio = prefix_count / no_of_derived_words
		suffix_derived_ratio = suffix_count / no_of_derived_words
		total_affix_derived_ratio = (prefix_count + suffix_count) / no_of_derived_words

	suffix_token_ratio = suffix_count / word_count
	total_affix_token_ratio = (prefix_count + suffix_count) / word_count

	return (
		prefix_token_ratio, prefix_derived_ratio, suffix_token_ratio,
		suffix_derived_ratio, total_affix_token_ratio, total_affix_derived_ratio
	)


# ----------------------------------------------------------------------------------

# INFLECTIONAL MORPHOLOGY - FOCUS FEATURES

def get_inflectional_morph(text):
	splitted = re.split('[?.]+', text)
	splitted = [i for i in splitted if i]   #removes empty strings in list
	tokenized_split = [wordpunct_tokenize(i.strip()) for i in splitted]
	tagged_text = pos_tagger.tag_sents(tokenized_split)

	actor_verbs = 0
	object_verbs = 0
	benefactive_verbs = 0
	locative_verbs = 0
	instrumental_verbs = 0
	referential_verbs = 0
	infinitive_verbs = 0
	present_participle_verbs = 0
	future_participle_verbs = 0
	past_participle_verbs = 0
	recent_past_verbs = 0
	aux_verbs = 0


	for sentence in tagged_text:
		for word in sentence:
			if '|' not in word[0]:
				pos = word[1].split('|')[1]
				if pos == 'VBAF':					# actor_focus_ratio
					actor_verbs += 1
				if pos == 'VBOF':
					object_verbs += 1				# object_focus_ratio
				if pos == 'VBOB':
					benefactive_verbs += 1			# benefactive_focus_ratio
				if pos == 'VBOL':
					locative_verbs += 1				# locative_focus_ratio
				if pos == 'VBOI':
					instrumental_verbs += 1			# instrumental focus_ratio
				if pos == 'VBRF':
					referential_verbs += 1			# referencial focus_ratio
				if pos == 'VBW':
					infinitive_verbs += 1			# inf verb_ratio
				if pos == 'VBTS':
					past_participle_verbs += 1		# perfective_verb_ratio*
				if pos == 'VBTR':
					present_participle_verbs += 1	# imperfective_verb_ratio*
				if pos == 'VBTF':
					future_participle_verbs += 1	# contemplative_verb_ratio*		add all with * for participle verb ratio
				if pos == 'VBTP':
					recent_past_verbs += 1			# recent_past_verb ratio
				if pos == 'VBS':
					aux_verbs += 1					# aux_verb_ratio
				
	if word_count == 0:
		tuple_of_zeroes = (0,) * 13
		return tuple_of_zeroes
	
	actor_focus_ratio = actor_verbs/word_count
	object_focus_ratio = object_verbs/word_count
	benefactive_focus_ratio = benefactive_verbs/word_count
	locative_focus_ratio = locative_verbs/word_count
	instrumental_focus_ratio = instrumental_verbs/word_count
	referential_focus_ratio = referential_verbs/word_count
	infinitive_verb_ratio = infinitive_verbs/word_count
	participle_verb_ratio = (past_participle_verbs + present_participle_verbs + future_participle_verbs)/word_count
	perfective_verb_ratio = past_participle_verbs/word_count
	imperfective_verb_ratio = present_participle_verbs/word_count
	contemplative_verb_ratio = future_participle_verbs/word_count
	recent_past_verb_ratio = recent_past_verbs/word_count
	aux_verb_ratio = aux_verbs/word_count

	return (
		actor_focus_ratio, object_focus_ratio, benefactive_focus_ratio,
		locative_focus_ratio, instrumental_focus_ratio, referential_focus_ratio,
		infinitive_verb_ratio, participle_verb_ratio, perfective_verb_ratio,
		imperfective_verb_ratio, contemplative_verb_ratio, recent_past_verb_ratio,
		aux_verb_ratio
	)

# DIRECTORIES FROM train.py when calling 'train.py'
os.environ['JAVAHOME'] = java_path
modelfile = stanford_dir + "/models/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger"
jarfile = stanford_dir + "/stanford-postagger.jar"

pos_tagger=StanfordPOSTagger(modelfile,jarfile,java_options="-Xmx8G")		# Change -Xmx4G to -XmxYG as needed where Y is the heap size in Gigabytes

if __name__ == "__main__":
	import time
	import tracemalloc
	text = "Ito ang halimaw ng mga kulay.  Ngayong araw  gumising siyang kakaiba ang pakiramdam  nalilito  tuliro… Hindi niya alam kung ano ang mali sa kaniya.  Nalilito ka na naman? Hindi ka na natuto.  Anong gulo ang ginawa mo sa iyong mga damdamin!"

	start = time.time()
	tracemalloc.start()
	print('get_derivational_morph:', get_derivational_morph(text))
	for i in get_inflectional_morph(text):
		print(i)
	print(f'\nTIME: {time.time() - start}s. SPACE: {tracemalloc.get_traced_memory()}.')
	tracemalloc.stop()