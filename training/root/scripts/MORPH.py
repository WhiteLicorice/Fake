from nltk.tag import StanfordPOSTagger
from nltk import word_tokenize
from root.scripts.TRAD import word_count_per_doc, sentence_count_per_doc, cleaner
import os, math, re


from nltk import data as nltk_data
nltk_data.path.append("./root/runtime_env/nltk_data")

# MAIN FUNCTIONS


# ----------------------------------------------------------------------------------
# DERIVATIONAL MORPHOLOGY 

def get_derivational_morph(text):

	prefix_count = 0
	suffix_count = 0
	derived_words = []
	
	word_count = word_count_per_doc(text)

	prefix_list = ['ma','pa','na','hing','i','ika','in','ipa','ipag','ipang','ka','ma','mag','magka','magpa','maka','maki','makipag','mang','mapag','may','napaka','pag','pagka','pagkaka','paki','pakikipag','pala','pam','pan','pang','pinaka','tag','taga','tagapag','um']
	suffix_list = ['an','ay','ng','oy','in','ing']

	splitted = re.split('[!?.]+', text)
	splitted = [i for i in splitted if i]

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

	if len(derived_words) == 0:
		prefix_derived_ratio = 0
		suffix_derived_ratio = 0
		total_affix_derived_ratio = 0

	else:
		prefix_derived_ratio = prefix_count / len(derived_words)
		suffix_derived_ratio = suffix_count / len(derived_words)
		total_affix_derived_ratio = (prefix_count + suffix_count) / len(derived_words)

	suffix_token_ratio = suffix_count / word_count
	total_affix_token_ratio = (prefix_count + suffix_count) / word_count

	return [prefix_token_ratio, prefix_derived_ratio, suffix_token_ratio, suffix_derived_ratio, total_affix_token_ratio,total_affix_derived_ratio]


# ----------------------------------------------------------------------------------

# INFLECTIONAL MORPHOLOGY - FOCUS FEATURES
def actor_focus_ratio(text):
	splitted = re.split('[?.]+', text)
	splitted = [i for i in splitted if i]   #removes empty strings in list

	word_count = word_count_per_doc(text)

	verb_counter = 0
	actor_verbs = 0
	for i in splitted:
		i = i.strip()
		tagged_text = pos_tagger.tag(word_tokenize(i))
		for x in tagged_text:
			if '|' not in x[0]:
				pos = x[1].split('|')[1]
				#print(pos)
				if pos[:2] == 'VB':
					verb_counter += 1
				if pos == 'VBAF':
					actor_verbs += 1

	if word_count == 0:
		return 0

	return (actor_verbs/word_count)


def object_focus_ratio(text):
	splitted = re.split('[?.]+', text)
	splitted = [i for i in splitted if i]   #removes empty strings in list

	word_count = word_count_per_doc(text)

	verb_counter = 0
	object_verbs = 0
	for i in splitted:
		i = i.strip()
		tagged_text = pos_tagger.tag(word_tokenize(i))
		for x in tagged_text:
			if '|' not in x[0]:
				pos = x[1].split('|')[1]
				#print(pos)
				if pos[:2] == 'VB':
					verb_counter += 1
				if pos == 'VBOF':
					object_verbs += 1

	if word_count == 0:
		return 0

	return (object_verbs/word_count)


def benefactive_focus_ratio(text):
	splitted = re.split('[?.]+', text)
	splitted = [i for i in splitted if i]   #removes empty strings in list

	word_count = word_count_per_doc(text)

	verb_counter = 0
	benefactive_verbs = 0
	for i in splitted:
		i = i.strip()
		tagged_text = pos_tagger.tag(word_tokenize(i))
		for x in tagged_text:
			if '|' not in x[0]:
				pos = x[1].split('|')[1]
				#print(pos)
				if pos[:2] == 'VB':
					verb_counter += 1
				if pos == 'VBOB':
					benefactive_verbs += 1

	if word_count == 0:
		return 0

	return (benefactive_verbs/word_count)


def locative_focus_ratio(text):
	splitted = re.split('[?.]+', text)
	splitted = [i for i in splitted if i]   #removes empty strings in list

	word_count = word_count_per_doc(text)

	verb_counter = 0
	locative_verbs = 0
	for i in splitted:
		i = i.strip()
		tagged_text = pos_tagger.tag(word_tokenize(i))
		for x in tagged_text:
			if '|' not in x[0]:
				pos = x[1].split('|')[1]
				#print(pos)
				if pos[:2] == 'VB':
					verb_counter += 1
				if pos == 'VBOL':
					locative_verbs += 1

	if word_count == 0:
		return 0

	return (locative_verbs/word_count)


def instrumental_focus_ratio(text):
	splitted = re.split('[?.]+', text)
	splitted = [i for i in splitted if i]   #removes empty strings in list

	word_count = word_count_per_doc(text)

	verb_counter = 0
	instrumental_verbs = 0
	for i in splitted:
		i = i.strip()
		tagged_text = pos_tagger.tag(word_tokenize(i))
		for x in tagged_text:
			if '|' not in x[0]:
				pos = x[1].split('|')[1]
				#print(pos)
				if pos[:2] == 'VB':
					verb_counter += 1
				if pos == 'VBOI':
					instrumental_verbs += 1

	if word_count == 0:
		return 0

	return (instrumental_verbs/word_count)


def referential_focus_ratio(text):
	splitted = re.split('[?.]+', text)
	splitted = [i for i in splitted if i]   #removes empty strings in list

	word_count = word_count_per_doc(text)

	verb_counter = 0
	referential_verbs = 0
	for i in splitted:
		i = i.strip()
		tagged_text = pos_tagger.tag(word_tokenize(i))
		for x in tagged_text:
			if '|' not in x[0]:
				pos = x[1].split('|')[1]
				#print(pos)
				if pos[:2] == 'VB':
					verb_counter += 1
				if pos == 'VBRF':
					referential_verbs += 1

	if word_count == 0:
		return 0

	return (referential_verbs/word_count)

# ----------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------
# INFLECTIONAL MORPHOLOGY - TENSE FEATURES

def infinitive_verb_ratio(text):
	splitted = re.split('[?.]+', text)
	splitted = [i for i in splitted if i]   #removes empty strings in list

	word_count = word_count_per_doc(text)

	verb_counter = 0
	infinitive_verbs = 0
	for i in splitted:
		i = i.strip()
		tagged_text = pos_tagger.tag(word_tokenize(i))
		for x in tagged_text:
			if '|' not in x[0]:
				pos = x[1].split('|')[1]
				#print(pos)
				if pos[:2] == 'VB':
					verb_counter += 1
				if pos == 'VBW':
					infinitive_verbs += 1

	if word_count == 0:
		return 0

	return (infinitive_verbs/word_count)


def participle_verb_ratio(text):
	splitted = re.split('[?.]+', text)
	splitted = [i for i in splitted if i]   #removes empty strings in list

	word_count = word_count_per_doc(text)

	verb_counter = 0
	past_participle_verbs = 0
	present_participle_verbs = 0
	future_participle_verbs = 0
	for i in splitted:
		i = i.strip()
		tagged_text = pos_tagger.tag(word_tokenize(i))
		for x in tagged_text:
			if '|' not in x[0]:
				pos = x[1].split('|')[1]
				#print(pos)
				if pos[:2] == 'VB':
					verb_counter += 1
				if pos == 'VBTS':
					past_participle_verbs += 1
				if pos == 'VBTR':
					present_participle_verbs += 1
				if pos == 'VBTF':
					future_participle_verbs += 1

	if word_count == 0:
		return 0

	return ((past_participle_verbs+present_participle_verbs+future_participle_verbs)/word_count)

def perfective_verb_ratio(text):
	splitted = re.split('[?.]+', text)
	splitted = [i for i in splitted if i]   #removes empty strings in list

	word_count = word_count_per_doc(text)

	verb_counter = 0
	past_participle_verbs = 0
	for i in splitted:
		i = i.strip()
		tagged_text = pos_tagger.tag(word_tokenize(i))
		for x in tagged_text:
			if '|' not in x[0]:
				pos = x[1].split('|')[1]
				#print(pos)
				if pos[:2] == 'VB':
					verb_counter += 1
				if pos == 'VBTS':
					past_participle_verbs += 1

	if word_count == 0:
		return 0

	return (past_participle_verbs/word_count)


def imperfective_verb_ratio(text):
	splitted = re.split('[?.]+', text)
	splitted = [i for i in splitted if i]   #removes empty strings in list

	word_count = word_count_per_doc(text)

	verb_counter = 0
	present_participle_verbs = 0
	for i in splitted:
		i = i.strip()
		tagged_text = pos_tagger.tag(word_tokenize(i))
		for x in tagged_text:
			if '|' not in x[0]:
				pos = x[1].split('|')[1]
				#print(pos)
				if pos[:2] == 'VB':
					verb_counter += 1
				if pos == 'VBTR':
					present_participle_verbs += 1

	if word_count == 0:
		return 0

	return (present_participle_verbs/word_count)


def contemplative_verb_ratio(text):
	splitted = re.split('[?.]+', text)
	splitted = [i for i in splitted if i]   #removes empty strings in list

	word_count = word_count_per_doc(text)

	verb_counter = 0
	future_participle_verbs = 0
	for i in splitted:
		i = i.strip()
		tagged_text = pos_tagger.tag(word_tokenize(i))
		for x in tagged_text:
			if '|' not in x[0]:
				pos = x[1].split('|')[1]
				#print(pos)
				if pos[:2] == 'VB':
					verb_counter += 1
				if pos == 'VBTF':
					future_participle_verbs += 1

	if word_count == 0:
		return 0

	return (future_participle_verbs/word_count)


def recent_past_verb_ratio(text):
	splitted = re.split('[?.]+', text)
	splitted = [i for i in splitted if i]   #removes empty strings in list

	word_count = word_count_per_doc(text)

	verb_counter = 0
	recent_past_verbs = 0
	for i in splitted:
		i = i.strip()
		tagged_text = pos_tagger.tag(word_tokenize(i))
		for x in tagged_text:
			if '|' not in x[0]:
				pos = x[1].split('|')[1]
				#print(pos)
				if pos[:2] == 'VB':
					verb_counter += 1
				if pos == 'VBTP':
					recent_past_verbs += 1

	if word_count == 0:
		return 0

	return (recent_past_verbs/word_count)


def aux_verb_ratio(text):
	splitted = re.split('[?.]+', text)
	splitted = [i for i in splitted if i]   #removes empty strings in list

	word_count = word_count_per_doc(text)

	verb_counter = 0
	aux_verbs = 0
	for i in splitted:
		i = i.strip()
		tagged_text = pos_tagger.tag(word_tokenize(i))
		for x in tagged_text:
			if '|' not in x[0]:
				pos = x[1].split('|')[1]
				#print(pos)
				if pos[:2] == 'VB':
					verb_counter += 1
				if pos == 'VBS':
					aux_verbs += 1

	if word_count == 0:
		return 0

	return (aux_verbs/word_count)



#UTILITY FUNCTIONS
#returns the T
def unique_tokentype_identifier(text):
	tagged_text = pos_tagger.tag([text.lower()])
	unique_tokens = []

	for i in tagged_text:
		if '|' not in i[0]:
			pos = i[1].split('|')[1]
			if pos not in unique_tokens:
				unique_tokens.append(pos)

	return unique_tokens


# DIRECTORIES FROM train.py when calling 'train.py'
java_path = "./root/runtime_env/custom_java/bin"

os.environ['JAVAHOME'] = java_path

stanford_dir = "./root/runtime_env/stanford-postagger-full-2020-11-17"

modelfile = stanford_dir + "/models/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger"
jarfile = stanford_dir + "/stanford-postagger.jar"

pos_tagger=StanfordPOSTagger(modelfile,jarfile,java_options="-Xmx8G")		# Change -Xmx4G to -XmxYG as needed where Y is the heap size in Gigabytes