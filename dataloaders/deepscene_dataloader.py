import os
import re
import numpy as np
import dataloaders.transforms as transforms

from imageio import imread
from torch.utils.data import Dataset, DataLoader

to_tensor = transforms.ToTensor()

iheight, iwidth = 472, 872 # original image size (there is some variation in DeepScene dataset)

class DeepSceneDataset(Dataset):
	def __init__(self, root, type='train', train_extra=True):
		self.root = root
		self.output_size = (224, 224) #(224, 448)

		# search for images
		self.rgb_files, self.depth_files = self.gather_images(os.path.join(root, 'rgb'),
											         os.path.join(root, 'depth_gray'))

		if type == 'train' and train_extra:
			extra_root = root + 'extra'
			extra_rgb, extra_depth = self.gather_images(os.path.join(extra_root, 'rgb'),
											    os.path.join(extra_root, 'depth_gray'))
 
		if len(self.rgb_files) == 0:
			raise (RuntimeError("Empty dataset - found no image pairs under \n" + root))

		# determine if 16-bit or 8-bit depth images
		self.depth_16 = False

		if imread(self.depth_files[0]).dtype.type is np.uint16: 
			self.depth_16 = True
			self.depth_16_max = 5000 #20000

		print('found {:d} image pairs with {:s}-bit depth under {:s}'.format(len(self.rgb_files), "16" if self.depth_16 else "8", root))

		# setup transforms
		if type == 'train':
			self.transform = self.train_transform
		elif type == 'val':
			self.transform = self.val_transform
		else:
			raise (RuntimeError("Invalid dataset type: " + type + "\n"
				       		"Supported dataset types are: train, val"))

	def gather_images(self, images_path, labels_path):
		def sorted_alphanumeric(data):
			convert = lambda text: int(text) if text.isdigit() else text.lower()
			alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
			return sorted(data, key=alphanum_key)

		#print('searching for images under: ')
		#print('    ' + images_path)
		#print('    ' + labels_path)

		image_files = sorted_alphanumeric(os.listdir(images_path))
		label_files = sorted_alphanumeric(os.listdir(labels_path))

		if len(image_files) != len(label_files):
			print('warning:  images path has a different number of files than labels path')
			print('   ({:d} files) - {:s}'.format(len(image_files), images_path))
			print('   ({:d} files) - {:s}'.format(len(label_files), labels_path))
			
		for n in range(len(image_files)):
			image_files[n] = os.path.join(images_path, image_files[n])
			label_files[n] = os.path.join(labels_path, label_files[n])
			
			#print('{:s} -> {:s}'.format(image_files[n], label_files[n]))

		return image_files, label_files

	def train_transform(self, rgb, depth):
		s = np.random.uniform(1.0, 1.5) # random scaling
		depth_np = depth #/ s
		angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
		do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

		# perform 1st step of data augmentation
		transform = transforms.Compose([
			#transforms.Resize(240.0 / iheight), # this is for computational efficiency, since rotation can be slow
			#transforms.Rotate(angle),
			#transforms.Resize(s),
			#transforms.CenterCrop(self.output_size),
			#transforms.HorizontalFlip(do_flip)
			transforms.Resize(self.output_size)
		])

		rgb_np = transform(rgb)
		#rgb_np = self.color_jitter(rgb_np) # random color jittering
		rgb_np = np.asfarray(rgb_np, dtype='float') / 255

		depth_np = transform(depth_np)
		depth_np = np.asfarray(depth_np, dtype='float')

		if self.depth_16:
			depth_np = depth_np / self.depth_16_max
		else:
			depth_np = depth_np / 255

		return rgb_np, depth_np

	def val_transform(self, rgb, depth):
		depth_np = depth

		transform = transforms.Compose([
			#transforms.Resize(240.0 / iheight),
			#transforms.CenterCrop(self.output_size),
			transforms.Resize(self.output_size)
		])

		rgb_np = transform(rgb)
		rgb_np = np.asfarray(rgb_np, dtype='float') / 255

		depth_np = transform(depth_np)
		depth_np = np.asfarray(depth_np, dtype='float')

		if self.depth_16:
			depth_np = depth_np / self.depth_16_max
		else:
			depth_np = depth_np / 255

		return rgb_np, depth_np

	def load_rgb(self, index):
		return imread(self.rgb_files[index], as_gray=False, pilmode="RGB")

	def load_depth(self, index):
		if self.depth_16:
			depth = imread(self.depth_files[index])
			depth[depth == 65535] = 0	# map 'invalid' to 0
			return depth
		else:		
			depth = imread(self.depth_files[index], as_gray=False, pilmode="L")
			#depth[depth == 0] = 255       # map 0 -> 255
			return depth

	def __len__(self):
		return len(self.rgb_files)

	def __getitem__(self, index):
		rgb = self.load_rgb(index)
		depth = self.load_depth(index)

		#print(self.rgb_files[index] + str(rgb.shape))
		#print(self.depth_files[index] + str(depth.shape))
		#print(depth)

		# apply train/val transforms
		if self.transform is not None:
			rgb_np, depth_np = self.transform(rgb, depth)
		else:
			raise(RuntimeError("transform not defined"))

		# convert from numpy to torch tensors
		input_tensor = to_tensor(rgb_np)

		while input_tensor.dim() < 3:
			input_tensor = input_tensor.unsqueeze(0)

		depth_tensor = to_tensor(depth_np)
		depth_tensor = depth_tensor.unsqueeze(0)

		#print("{:04d} rgb =   ".format(index) + str(input_tensor.shape))
		#print("{:04d} depth = ".format(index) + str(depth_tensor.shape))

		return input_tensor, depth_tensor

