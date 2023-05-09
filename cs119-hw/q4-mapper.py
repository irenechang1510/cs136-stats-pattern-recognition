#Mapper:
#!/usr/bin/python 
import sys, re 
import os
import requests 
import json
import urllib
def main(argv):
	president_score = {}
	url = "https://raw.githubusercontent.com/fnielsen/afinn/master/afinn/data/AFINN-en-165.txt"
	file = urllib.request.urlopen(url)
	afinn = dict()
	for line in file:
		decoded_line = line.decode("utf-8")
		split = decoded_line.split('\t')
		afinn[split[0]] = int(split[1].strip())
	line = sys.stdin.readline()

	pattern = re.compile("[a-zA-Z][a-zA-Z0-9]*") 
	try:
		while line:

			filepath = os.getenv('map_input_file') # get the filepath 

			filename = os.path.split(filepath)[-1]
			president = filename.split('_')[0]
			for word in pattern.findall(line):
				if afinn.has_key(word): 
					score = afinn[word]
				else:
					score = 0
			if president_score.has_key(president): 
				president_score[president] += score
			else:
				president_score[president] = score
			line = sys.stdin.readline() 
	except "end of file":
		return None
	for key, value in president_score.items():
		print(key + "\t" + str(value))

if __name__ == "__main__":
    main(sys.argv)


