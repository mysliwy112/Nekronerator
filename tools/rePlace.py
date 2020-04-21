from os import listdir
import random as rd
import copy

datadir="../dataset/"

dirs=["base/","mirror/"]

levelbase= [['122' for x in range(60)] for y in range(30)] 

iterall=0
for dir in dirs:
	for filename in listdir(datadir+dir):
		data=[]

		with open(datadir+dir+filename,"r") as file:
			data=file.readlines()

		data=[line.replace('\n','') for line in data]
		data=[line.split(" ") for line in data]

		width=int(data[0][0])
		height=int(data[0][1])
		
		level=[]
		
		for line in range(2,height+2):
			level.append(data[line][0:width])
		#print(level)
		#input()
		
		y=rd.randrange(1,4)
		while y<30-height:
			x=rd.randrange(1,6)
			while x<60-width:
				newlevel=copy.deepcopy(levelbase)
				for line in range(0,height):
					#print(line)
					#print(newlevel[line+y])
					#input()
					newlevel[line+y][x:x+width]=level[line][0:width]
					#print(newlevel[line+y])
					#input()

				with open(datadir+"replace/"+str(iterall)+"x="+str(x)+"-y="+str(y)+"-"+filename,"w") as file:
					iterall+=1;
					file.write(data[0][0]+" "+data[0][1]+" "+data[0][2]+"\n")
					file.write("{:03d} ".format(int(data[1][0])+x)+"{:03d}\n".format(int(data[1][1])+y))

					for line in newlevel:
						for node in range(len(line)):
							file.write(line[node])
							if node!=len(line)-1:
								file.write(" ")
						file.write('\n')
				x+=rd.randrange(1,6)
			y+=rd.randrange(1,4)
