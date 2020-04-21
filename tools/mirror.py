from os import listdir

for filename in listdir("../dataset"):

	data=[]

	with open("../dataset/"+filename,"r") as file:
		data=file.readlines()

	data=[line.replace('\n','') for line in data]
	data=[line.split(" ") for line in data]

	width=int(data[0][0])
	height=int(data[0][1])
	data[1][0]="{:03d}".format(width-int(data[1][0])-1)

	for y in range(2, height+2):
		for x in range(0,int(width/2)):
			data[y][x],data[y][width-x-1]=data[y][width-x-1],data[y][x]

	with open("../mirror/"+filename,"w") as file:
		for line in data[0:2]:
			for node in range(len(line)):
				file.write(line[node])
				if node!=len(line)-1:
					file.write(" ")
			file.write('\n')

		for line in data[2:]:
			for node in range(len(line)):
				if line[node] == '022':
					file.write('024')
				elif line[node] == '024':
					file.write('022')
				else:
					file.write(line[node])
				if node!=len(line)-1:
					file.write(" ")
			file.write('\n')
