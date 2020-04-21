from os import listdir
import random as rd

i=0
for filename in listdir("../in"):
	orig=[]
	data=[]
	with open("../in/"+filename,"r") as file:
		orig=file.readlines()
		
	data=[line.split(" ") for line in orig]
		
	width=60
	height=30
	herosX=0
	herosY=0
	for a in range(len(data)):
		for b in range(len(data[a])):
			if data[a][b]=='015':
				herosX=b
				herosY=a
				break
				
	
	with open("../out/"+"BD_CLEV"+str(i)+".FLD","w") as file:
		file.write("{:03d} ".format(width))
		file.write("{:03d} ".format(height))
		file.write("{:03d}\n".format(rd.randrange(0,5)))
		
		file.write("{:03d} ".format(herosX))
		file.write("{:03d}\n".format(herosY))
		for line in orig:
			file.write(line)
	i+=1