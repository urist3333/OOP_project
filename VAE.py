import tensorflow as tf
from Encoder import Encoder
from Decoder import Decoder
from losses import kl_divergence,log_diag_mvn
class VAE(tf.keras.Model):
    """
    Variational Autoencoder (VAE) CLASS
    This class inherits from tf.keras.Model
    The VAE uses an encoder and decoder class to both encode and decode 
    and compute the loss after each iteration of trainning.
    The loss computed is the Evidence Lower Bound (ELBO)

    Args:
        Keras networks: encoder_network, decoder_network
    Attributes: 
        vae_loss: Tensor, only set after first call()
        encoder: Encoder Object
        decoder: Decoder Object
    """
    
    def __init__(self, encoder_network, decoder_network):
        
        super().__init__()

        self.encoder = Encoder(encoder_network)
        self.decoder = Decoder(decoder_network)

    @property
    def latent_dim(self):
        """
        Decorator method .latent_dim 
        Used for dynamically forwarding a value from encoder, useful for getting dimensions to plot latent_dim
        """
        return self.encoder.latent_dim

    def call(self, x):
        """
        Overrides call() from tf.keras.Model 
        Uses outputs from encoder and decoder objects to compute ELBO
        Returns the negative ELBO
        """
        z,z_mu, z_logvar = self.encoder(x)
        _,mu_x,x_logsigma = self.decoder(z)
        kl_div = kl_divergence(z_mu,z_logvar)
        logp = log_diag_mvn(x,mu_x,x_logsigma)
        
        elbo = logp -kl_div
        self.vae_loss = -tf.reduce_mean(elbo)
        return self.vae_loss
    
    
    @tf.function
    def train(self, x, optimizer):
        """
        Training function
        Uses a decorator @tf.function to enhance training perfomance.
        Returns the loss for visualizations or logging progress
        """
        with tf.GradientTape() as tape:
            loss = self.call(x)
        gradients = tape.gradient(self.vae_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss 
    def sample_z(self,x):
        """
        Helper method
        Samples a noisy z from the latent space. Useful for visualizing the latent space
        """
        z,_,_ = self.encoder(x)
        return z
    def sample_zmean(self,x):
        """
        Helper method 
        Samples a mu_z from the latent space. Useful for visualizing the latent space
        """
        _,mu_z,_ = self.encoder(x)
        return mu_z
    
    
    def reconstruct_mean(self,z):
        """
        Helper method
        Samples a reconstruction mu_x. Useful for visualizing sharper reconstructions 
        """
        _,mu_x,_ = self.decoder(z)
        return mu_x
    def reconstruct_noisy(self,z):
        """
        Helper method
        Samples a reconstruction xhat. Useful for visualizing noisy reconstructions
        """
        xhat,_,_ = self.decoder(z)    
        return xhat
    
    
   
        


