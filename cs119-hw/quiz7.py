# QUESTION 1
# from nltk.corpus import stopwords
# import requests
# stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
# stopwords = set(stopwords_list.decode().splitlines()) 
# stopwords = list(stopwords)

# def preprocess(doc):
# 	list_words = doc.split(" ")
# 	# remove stop words and lowercase
# 	# stop_words = list(stopwords.words('english'))
# 	list_words = [word.lower() for word in list_words if word not in stopwords]	

# 	# re-join with spaces
# 	processed = ' '.join(list_words)
# 	return processed

# def shingles(doc, n):
# 	shingling = set()
# 	for i in range(len(doc) - n + 1):
# 		shingling.add(doc[i:i+n])
# 	return shingling

# def jaccard(sh_set_1, sh_set_2):
# 	set_union = sh_set_1.union(sh_set_2)
# 	set_intersection = sh_set_1.intersection(sh_set_2)
# 	return len(set_intersection)/float(len(set_union))

# if __name__ == '__main__':
# 	doc1 = 'Life is suffering'
# 	doc2 = 'Suffering builds character'
# 	doc3 = 'Character is the essence of life'

# 	doc1 = preprocess(doc1)
# 	doc2 = preprocess(doc2)
# 	doc3 = preprocess(doc3)

# 	sh_set1 = shingles(doc1, 2)
# 	sh_set2 = shingles(doc2, 2)
# 	sh_set3 = shingles(doc3, 2)

# 	print("Jaccard between doc1 and doc2: {}".format(jaccard(sh_set1, sh_set2)))
# 	print("Jaccard between doc2 and doc3: {}".format(jaccard(sh_set2, sh_set3)))
# 	print("Jaccard between doc3 and doc1: {}".format(jaccard(sh_set3, sh_set1)))

# QUESTION 3
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.data import load

boston_year = "My first week in Cambridge a car full of white boys \
tried to run me off the road, and spit through the window, \
open to ask directions. I was always asking directions \
and always driving: to an Armenian market \
in Watertown to buy figs and string cheese, apricots, \
dark spices and olives from barrels, tubes of paste \
with unreadable Arabic labels. I ate \
stuffed grape leaves and watched my lips swell in the mirror. \
The floors of my apartment would never come clean. \
Whenever I saw other colored people \
in bookshops, or museums, or cafeterias, I'd gasp, \
smile shyly, but they'd disappear before I spoke. \
What would I have said to them? Come with me? Take \
me home? Are you my mother? No. I sat alone \
in countless Chinese restaurants eating almond \
cookies, sipping tea with spoons and spoons of sugar. \
Popcorn and coffee was dinner. When I fainted \
from migraine in the grocery store, a Portuguese \
man above me mouthed: 'No breakfast.' He gave me \
orange juice and chocolate bars. The color red \
sprang into relief singing Wagner's Walküre. \
Entire tribes gyrated and drummed in my head. \
I learned the samba from a Brazilian man \
so tiny, so festooned with glitter I was certain \
that he slept inside a filigreed, Fabergé egg. \
No one at the door: no salesmen, Mormons, meter \
readers, exterminators, no Harriet Tubman, \
no one. Red notes sounding in a grey trolley town."

list_words = boston_year.split(" ")
list_words = [word.lower() for word in list_words]
reconstruct = " ".join(list_words)

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
sent_text = nltk.sent_tokenize(reconstruct) # this gives us a list of sentences

# now loop over each sentence and tokenize it separately
all_tagged = [nltk.pos_tag(nltk.word_tokenize(sent)) for sent in sent_text]

# dictionary of POS
tagdict = load('help/tagsets/upenn_tagset.pickle')

from collections import defaultdict
pos_dict = defaultdict(list)

for sent in all_tagged:
	for word in sent:
		tag = word[1][:2]
		if word[0] not in pos_dict[tag]:
			pos_dict[tag].append(word[0])

# for i in pos_dict:
# 	print(i)
# 	print(pos_dict[i])
# 	print("\n")
data = []
data.append((boston_year, dict(pos_dict)))

from pyspark import SparkContext,SparkConf
from pyspark.sql import SQLContext
sc = SparkContext()
spark = SQLContext(sc)

df = spark.createDataFrame(data, ["Poem", "word_dict"])

from pyspark.sql.functions import explode,map_keys,col
keysDF = df.select(explode(map_keys(df.word_dict))).distinct()
keysList = keysDF.rdd.map(lambda x:x[0]).collect()
keyCols = list(map(lambda x: col("word_dict").getItem(x).alias(str(x)), keysList))
pandas_df = df.select(df.Poem, *keyCols).toPandas()
print(pandas_df[['Poem', 'NN', 'VB', 'JJ']])
