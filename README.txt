Command-Line Arguments
Dataset choice and visualization 
---------------------------------------------------------------------------------------------------------
Argument	                    Default	            Description
--dset	                        mnist_bw	        Choose dataset.
--version	                    m1	                Dataset variant (ignored for mnist_bw).
--visualize_latent	            None  	            Plot the latent space.
--generate_from_prior	        None    	        Generate images via decoder from sampling prior.
--generate_from_posterior	    None                Generate images via decoder from posterior.


Training Hyperparameters
----------------------------------------------------------------------------------------------------------
Argument                        Default             Description
--epochs                        50                  Chose number of epochs for training
--batch_size                    256                 Chose batch_size for training
--learning_rate                 1e-3                Chose learning_rate for training

#Quality of Life 
----------------------------------------------------------------------------------------------------------
Argument                        Default             Description
--save_plot                     None                Save plot to folder /figs
--silent_mode                   False               If provided turns off printing of losses




Programs

Classes
----------------------------------------------------------------------------------------------------------
Name                Purpose                                                Class type
BiCoder             Blueprint for subclasses                               Superclass for Encoder, decoder
Encoder             Encode                                                 Subclass of BiCoder
Decoder             Decode                                                 Subclass of BiCoder
VAE                 Utilize Encoder Decoder                                Subclass of tf.keras.Models 
DataLoader          Load data for training/testing to folder \data         Superclass for MnistDataLoader
MnistDataLoader     Load Mnist Data for training/testing to folder \data   Subclass of DataLoader

Helper programs
-----------------------------------------------------------------------------------------------------------
Name                Purpose                                 Used in 
train_vae           Implement the project                   NA
plot.utils.py       Visualzations                           train_vae.py
nn.py               Network arcitecture                     network_selector.py
network_selector.py Select arcitecture based on dset        train_vae.py
losses.py           Compute terms in ELBO                   VAE.py (Class)

Folders
Name                Purpose                                 
\figs               Store figures
\data               Store data (created upon running MnistDataLoader)
