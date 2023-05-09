# reducer
#!/usr/bin/python import sys
def main(argv): 
	president_valence = {} 
	line = sys.stdin.readline()
	while line:
		president, totalscore = line.split("\t") #get data from mapper
		try:
			totalscore = int(totalscore)
		except ValueErrror: 
			continue
		if president_valence.has_key(president): 
			president_valence[president] += totalscore
		else:
			president_valence[president] = totalscore
		line = sys.stdin.readline()
	
	for key, value in president_valence.items():
		print("{}\t{}".format(key, value)) 

if __name__ == "__main__":
    main(sys.argv)
