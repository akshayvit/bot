from numpy import hstack
from numpy import zeros
from numpy import ones
import numpy as np
from numpy.random import rand
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
import librosa as li
import os

def define_discriminator(n_inputs=2):
	model = Sequential()
	model.add(Dense(14, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def define_generator(latent_dim, n_outputs=2):
	model = Sequential()
	model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
	model.add(Dense(n_outputs, activation='linear'))
	return model

def define_gan(generator, discriminator):
	discriminator.trainable = False
	model = Sequential()
	model.add(generator)
	model.add(discriminator)
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

def generate_real_samples(n):
        path=r"E:\\python\\sadw"
        X=[]
        for r,d,f in os.walk(path):
                for file in f:
                        at,sr=li.load(path+"\\"+file)
                        X.append(at)
        X=np.array(X)
        t=np.reshape(X,(14,1))
        t=np.concatenate(t[0],axis=0)
        t=np.transpose(t,axes=0)
        y = ones((14, 1))
        t=t.tolist()
        return t, y

def generate_latent_points(latent_dim, n):
	x_input = randn(latent_dim * n)
	x_input = x_input.reshape(n, latent_dim)
	return x_input

def generate_fake_samples(generator, latent_dim, n):
	x_input = generate_latent_points(latent_dim, n)
	X = generator.predict(x_input)
	y = zeros((n, 1))
	return X, y

def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):
	x_real, y_real = generate_real_samples(n)
	_, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
	x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
	_, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
	print(epoch, acc_real, acc_fake)
	pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')
	pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
	print(x_fake)
	pyplot.show()

def train(g_model, d_model, gan_model, latent_dim, n_epochs=100, n_batch=128, n_eval=20):
	half_batch = int(n_batch / 2)
	for i in range(n_epochs):
		x_real, y_real = generate_real_samples(half_batch)
		print(x_real)
		print(y_real)
		x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		d_model.train_on_batch(x_real, y_real)
		d_model.train_on_batch(x_fake, y_fake)
		x_gan = generate_latent_points(latent_dim, n_batch)
		y_gan = ones((n_batch, 1))
		gan_model.train_on_batch(x_gan, y_gan)
		if (i+1) % n_eval == 0:
			summarize_performance(i, g_model, d_model, latent_dim)

latent_dim = 5
discriminator = define_discriminator()
generator = define_generator(latent_dim)
gan_model = define_gan(generator, discriminator)
train(generator, discriminator, gan_model, latent_dim)
