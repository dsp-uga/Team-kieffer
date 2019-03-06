from PIL import Image

#reading the hash list of training files
fp = open('/home/mohammadreza_im/project2/train.txt') # Open file on read mode
train_list = fp.read().split("\n") # Create a list containing all lines
fp.close() 
train_list = list(filter(None, train_list))

#read the photos and masks of training set with steps (3 frames from each video) and resizing
k = 0
for i in range(0,(len(train_list))):
	mpath = '/home/mohammadreza_im/project2/masks/'+train_list[i]+'.png'
	m = Image.open(mpath)
	res_m = m.resize((290, 210))
	train_frames = 0
	for j in range(0,100,20):
		train_frames +=1
		path = '/home/mohammadreza_im/project2/data/data/'+train_list[i]+'/frame00'+'%0*d' % (2, j)+'.png'
		f = Image.open(path)
		res_f = f.resize((290, 210))
		s_path = '/home/mohammadreza_im/project2/unet-ggg/tf_unet/train/'+train_list[i]+'_'+'%0*d' % (5, k)+'.tif'
		m_path = '/home/mohammadreza_im/project2/unet-ggg/tf_unet/train/'+train_list[i]+'_'+'%0*d' % (5, k)+'_mask.tif'
		k += 1
		res_f.save(s_path)
		res_m.save(m_path)

