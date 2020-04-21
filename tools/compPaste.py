from os import listdir

datas=[];
for filename in listdir("../levels"):
	with open("../levels/"+filename,"r") as file:
		datas.append(file.read())

outs=[]


for data in datas:
	has=False
	for out in outs:
		if(out==data):
			has=True;
			break;
	if(has==False):
		outs.append(data)
print(len(outs))

i=0

for out in outs:
	with open("../levelsout/boulder_"+str(i)+".fld","w") as file:
		file.write(out)
		i+=1