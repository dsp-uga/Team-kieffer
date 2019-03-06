import os

for line in open("test.py"):
	filename = line.strip()
	os.system("bash -c \"cp ../data/"+filename+"/frame0000.jpg ../result/"++filename" \"")