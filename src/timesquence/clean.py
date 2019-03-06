from PIL import Image

# get each pixel from the file 
def get_mask_data(filename):
	im = Image.open("../masks/"+filename+".png")
	pix = im.load()
	width, height = im.size
	#print(width,height)
	a = []
	for i in range(width):
		for j in range(height):
			a.append(pix[i,j])
	return a

#conbine the envrionment information and form a longer time sequence data
def get_x_data(filename):
	
	data = {}
	for index in range(100):
		index_number = str(index)
		if len(index_number) == 1:
			index_number = "000"+index_number
		else:
			index_number = "00" + index_number
		im = Image.open("../data/"+filename+"/frame"+index_number+".png")
		pix = im.load()
		width, height = im.size
		for i in range(width):
			for j in range(height):
				pix_index = i * height + j
				int(pix[i,j])
				if pix_index in data:
					data[pix_index].append(str(pix[i,j]))
				else:
					data[pix_index] = [str(pix[i,j])]

	result  = []
	for i in range(width*height):
		tmp_result = []
		
		try:
			tmp_result = tmp_result +data[i-1-width]  #left
		except:
			tmp_result = tmp_result +["0"]*100
		
		try:
			tmp_result = tmp_result + data[i-width] #right
		except:
			tmp_result = tmp_result + ["0"]*100
		
		
		try:
			tmp_result = tmp_result + data[i+1-width]  #down
		except:
			tmp_result = tmp_result + ["0"]*100
		
		try:
			tmp_result = tmp_result + data[i-1] #up
		except:
			tmp_result = tmp_result + ["0"]*100
		tmp_result = tmp_result + data[i]  #own point

		try:
			tmp_result = tmp_result + data[i+1]  #left
		except:
			tmp_result = tmp_result + ["0"]*100
		
		
		try:
			tmp_result = tmp_result + data[i-1+width] #righ
		except:
			tmp_result = tmp_result + ["0"]*100
		
		try:
			tmp_result = tmp_result + data[i+width]  #down
		except:
			tmp_result = tmp_result + ["0"]*100
		
		try:
			tmp_result = tmp_result + data[i+1+width] #up
		except:
			tmp_result = tmp_result + ["0"]*100
		

		result.append(tmp_result)
	return result

def get_test_data():
	pass

#The main funtion to clean the data and form time frequence data
def main()

	for line in open("../test.txt"):
		print(line)
		filename = line.strip()
		#mask = get_mask_data(line.strip())
		data = get_x_data(line.strip())
		o_file = open("../env_label/"+filename+".csv",'w')

		for i in range(len(data)):
			o_file.write("%s\t%s\n"%("0",",".join(data[i])))
			#o_file.write("%s\t%s\n"%(mask[i],",".join(data[i])))
main()


