import json

def write(filename, data):
	with open(filename, 'w') as outfile:# overwrite existing
	#with open(filename, 'a') as outfile: # appending after
		json.dump(data, outfile)

def read(filename):
	with open(filename, 'r') as infile:
		return json.load(infile)
