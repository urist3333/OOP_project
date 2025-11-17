import tensorflow as tf
from BiCoder import BiCoder


class Encoder(BiCoder):
    """
    Encoder module used in variational autoencoders (VAEs).

    This class extends the `BiCoder` superclass and applies a neural
    network to map inputs into latent representations. It produces
    the mean (`mu_z`), log-variance (`logvar_z`), and a random sample `z`.

    Args:
        neural_net: A Keras model that outputs a latent representation
            

    Attributes:
        neural_net: The underlying neural network used to encode inputs.
        latent_dim: Integer representing the size of the latent representation. Determine on the first call.
    """   
    def __init__(self,neural_net): 
        super().__init__(neural_net) #referal to parent class
        #instance variables
        self.latent_dim = None


    def call(self,x):
        """
        call() method 
        Respects the contract and implements the call() method
        Perfoms the reparmeterization and returns a tuple of z, mu_z and log_var z
        Sets the latent_dim attribute once upon the first call
        """

        out = self.neural_net(x)
        if self.latent_dim is None:
            
            self.latent_dim = out.shape[1] // 2
        
        mu_z = out[:,:self.latent_dim]
        logvar_z = out[:,self.latent_dim:]
        sigma_z = tf.math.exp(0.5*logvar_z)

        eps = tf.random.normal(tf.shape(mu_z))

        z = mu_z +sigma_z*eps
        return z,mu_z, logvar_z 

   