from __future__ import division
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
from scipy.io.wavfile import write

start,end=[],[]

def define_discriminator(n_inputs=2):
	model = Sequential()
	model.add(Dense(50, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
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

def get_highest_energy(ts):
        global start,end
        y,max=[],-9999.9
        for i in range(0,len(ts)-100,100):
                st=ts[i:i+100]
                sum=0.0
                for  j in range(len(st)):
                        sum+=st[j]*st[j]
                if(sum>max):
                        y=st
                        start=ts[:i]
                        end=ts[i+100:]
                        max=sum
        return y
def generate_real_samples(n):
        path=r"E:\\python\\happyw"
        at=[]
        for r,d,f in os.walk(path):
                for file in f:
                        print(path+"\\"+file)
                        at,sr=li.load(path+"\\"+file, offset=15.0)
                        print(at,sr)
                        
        X=np.array(get_highest_energy(at))
        print(len(X))
        t=np.reshape(X,(50,2))
        print(t)
       # t=np.transpose(X,axes=0)
        y = ones((n, 1))
        #t=t.tolist()
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
        x_real, y_real = generate_real_samples(50)
        _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
        x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
        _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
        print(epoch, acc_real, acc_fake)
        pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')
        pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
        print(x_fake)
        generated_ats=[]
        generated_ats.extend(end)
        for i in range(len(x_fake)):
                generated_ats.append(x_fake[i][0])
                generated_ats.append(x_fake[i][1])
        print(generated_ats)
        generated_ats.extend(start)
        scaled = np.int16(generated_ats/np.max(np.abs(generated_ats)) * 32767)
        write('generated_audio.wav', 22050, scaled)
        pyplot.show()

def train(g_model, d_model, gan_model, latent_dim, n_epochs=200, n_batch=128, n_eval=20):
        half_batch = int(n_batch / 2)
        for i in range(0,n_epochs,n_eval):
                x_real, y_real = generate_real_samples(50)
                print(x_real)
                print(y_real)
                x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
                d_model.train_on_batch(x_real, y_real)
                d_model.train_on_batch(x_fake, y_fake)
                x_gan = generate_latent_points(latent_dim, n_batch)
                y_gan = ones((n_batch, 1))
                gan_model.train_on_batch(x_gan, y_gan)
                if (i) % n_eval == 0:
                        print(i)
                        summarize_performance(i, g_model, d_model, latent_dim)

latent_dim = 5
discriminator = define_discriminator()
generator = define_generator(latent_dim)
gan_model = define_gan(generator, discriminator)
train(generator, discriminator, gan_model, latent_dim)
