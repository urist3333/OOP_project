
import argparse

from VAE import VAE

import tensorflow as tf
import numpy as np
from MnistDataLoader import MnistDataLoader

from network_selecter import network_selecter
from plot_utils import plot_grid,plot_latent

def main():
    #__________________Dataset choice and visualization________________________________________
    parser = argparse.ArgumentParser(description="This program runs the train.py file")
    parser.add_argument("--dset", type= str, default= "mnist_bw")
    parser.add_argument("--version",type = str, default= "m1")
    parser.add_argument("--visualize_latent",action = "store_true")
    parser.add_argument("--generate_from_prior",action="store_true")
    parser.add_argument("--generate_from_posterior",action="store_true")
    parser.add_argument("--noisy", action ="store_true")
   

    #________________Trainning_Hyperparameters___________________________________
    parser.add_argument("--epochs",type = int,default = 50)
    parser.add_argument("--batch_size",type = int, default = 256)
    parser.add_argument("--learning_rate",type = float, default = 1e-3)

    #_________________Quality of life___________________________
    parser.add_argument("--save_plot",action = "store_true")
    parser.add_argument("--silent_mode", action="store_true", default=False)

  
   
        

    args = parser.parse_args()
   
    if args.dset =="mnist_bw":
        args.version = None

    #______________Training______________________________________________
    encoder_network, decoder_network = network_selecter(args.dset)
    model = VAE(encoder_network,decoder_network)
    optimizer = tf.keras.optimizers.Adam(learning_rate =args.learning_rate)
    my_data_loader = MnistDataLoader(dset = args.dset,version = args.version)
    tr_data = my_data_loader.get_training_data(batch_size=args.batch_size)

    for e in range(args.epochs):
        batch_loss = []
        for tr_batch in tr_data:
            loss = model.train(tr_batch,optimizer)
            batch_loss.append(loss)
        if not args.silent_mode:
            epoch_loss = tf.reduce_mean(batch_loss).numpy()
            print(f" Epoch: {e+1} | Loss = {epoch_loss} ")








    #___________Visualizing the latent space_______________________
    if args.visualize_latent:
        x_te = my_data_loader.get_testing_data()
        labels = my_data_loader.get_labels()
        if args.noisy:
            z = model.sample_z(x_te)
        else:
            z = model.sample_zmean(x_te)

        plot_latent(z,labels,args.dset,args.batch_size,args.epochs,"Latent",args.save_plot,args.noisy,args.learning_rate,args.version,)



    #____________Generating new image from prior dist.____________
    if args.generate_from_prior:
        latent_dim = model.latent_dim
        samples = 100
        z_prior = np.random.randn(samples, latent_dim).astype(np.float32)
        z_prior = tf.convert_to_tensor(z_prior)
        if args.noisy:    
            x_recon = model.reconstruct_noisy(z_prior)
    
        else:
            x_recon = model.reconstruct_mean(z_prior)
            
        plot_grid(x_recon,args.dset,args.batch_size,args.epochs,"Prior",args.save_plot,args.noisy,args.learning_rate,args.version)

    #__________Generating new image from posterior dist.____________
    if args.generate_from_posterior:
        te_data = my_data_loader.get_testing_data()
        z = model.sample_z(te_data)
        if args.noisy:   
            x_recon = model.reconstruct_noisy(z)
    
        else:
            x_recon = model.reconstruct_mean(z)
        plot_grid(x_recon,args.dset,args.batch_size,args.epochs,"Posterior",args.save_plot,args.noisy,args.learning_rate,args.version)


main()
import matplotlib.pyplot as plt
plt.show()




