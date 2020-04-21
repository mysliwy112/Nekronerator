import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import sys

from IPython import display

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


levels=5
training=True
if len(sys.argv)>1 and sys.argv[1]=="generate":
	training=False
	if len(sys.argv)>2:
		levels=int(sys.argv[2])









take =	{
  1:0,
  5:1,
  15:2,
  17:3,
  20:4,
  22:5,
  23:6,
  24:7,
  120:8,
  121:9,
  122:10,
  124:11,
  125:12
}

norm_val=(len(take)-1)/2


take_back = {v: k for k, v in take.items()}

data=[]

dirs=["./dataset/","./mirror/"]
for dir in dirs:
	list=os.listdir(dir)
	for filename in list:
		with open(dir+filename,"r") as file:
			d=file.readlines()[2:]
			d=[a.split(" ") for a in d]
			d=[[int(b) for b in a] for a in d]
			for a in range(0,2):
				d.append([122]*60)
			for a in d:
				for b in range(0,4):
					a.append(122)
			try:
				for a in range(len(d)):
					for b in range(len(d[a])):
						d[a][b]=take[d[a][b]]
			except KeyError as ero:
				raise Exception("File: "+dir+filename+" has unidentified number: "+str(ero.args[0]))
			data.append(d)


train_images=np.asarray(data)

train_images = train_images.reshape(train_images.shape[0], 32, 64, 1).astype('float32')
train_images = (train_images - norm_val) / norm_val # Normalize the images to [-1, 1]
BUFFER_SIZE = 60000
BATCH_SIZE = 64

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def make_generator_model():
	model = tf.keras.Sequential()
	model.add(layers.Dense(16*8*256, use_bias=False, input_shape=(100,)))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Reshape((8, 16, 256)))
	assert model.output_shape == (None, 8, 16, 256) # Note: None is the batch size

	model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
	assert model.output_shape == (None, 8, 16, 128)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
	assert model.output_shape == (None, 16, 32, 64)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
	assert model.output_shape == (None, 32, 64, 1)

	return model

def make_discriminator_model():
	model = tf.keras.Sequential()
	model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[32, 64, 1]))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))

	model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))

	model.add(layers.Flatten())
	model.add(layers.Dense(1))

	return model




generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')



discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
	real_loss = cross_entropy(tf.ones_like(real_output), real_output)
	fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
	total_loss = real_loss + fake_loss
	return total_loss

def generator_loss(fake_output):
	return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
								 discriminator_optimizer=discriminator_optimizer,
								 generator=generator,
								 discriminator=discriminator)


EPOCHS = 100000
noise_dim = 100
num_examples_to_generate = 8

seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
	noise = tf.random.normal([BATCH_SIZE, noise_dim])

	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		generated_images = generator(noise, training=True)

		real_output = discriminator(images, training=True)
		fake_output = discriminator(generated_images, training=True)

		gen_loss = generator_loss(fake_output)
		disc_loss = discriminator_loss(real_output, fake_output)

	gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
	gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

	generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
	for epoch in range(epochs):
		start = time.time()

		for image_batch in dataset:
			train_step(image_batch)

		# Produce images for the GIF as we go
		if (epoch + 1) % 50 == 0:
			display.clear_output(wait=True)
			generate_and_save_images(generator, epoch + 1, seed)

		# Save the model every 15 epochs
		if (epoch + 1) % 500 == 0:
			checkpoint.save(file_prefix = checkpoint_prefix)

		print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

def generate_and_save_images(model, epoch, test_input):
	predictions = model(test_input, training=False)

	fig = plt.figure(figsize=(4,4))

	unpredict=np.rint(predictions[:, :, :, 0] * norm_val + norm_val)
	with np.nditer(unpredict, op_flags=['readwrite']) as it:
		for x in it:
			x[...]=take_back.get(int(x),int(x))


	for i in range(unpredict.shape[0]):
		plt.subplot(2, 4, i+1)
		plt.imshow(unpredict[i,:,:], cmap='gray')
		plt.axis('off')
	np.savetxt('level_at_epoch_{:04d}.fld'.format(epoch),unpredict[0,:,:] ,fmt='%03d', delimiter=' ')

	plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
	#plt.show()
	plt.close()

def generate_levels(model, test_input):
	predictions = model(test_input, training=False)

	unpredict=np.rint(predictions[:, :, :, 0] * norm_val + norm_val)
	with np.nditer(unpredict, op_flags=['readwrite']) as it:
		for x in it:
			x[...]=take_back.get(int(x),int(x))


	for i in range(unpredict.shape[0]):
		np.savetxt('level_{:04d}.fld'.format(i),unpredict[0,:,:] ,fmt='%03d', delimiter=' ')


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

if training==True:
	train(train_dataset, EPOCHS)
else:
	train(train_dataset, 1)
	for i in range(levels):
		generate_levels(generator, tf.random.normal([1, noise_dim]))
