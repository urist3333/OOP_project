import tensorflow as tf
from BiCoder import BiCoder


class Decoder(BiCoder):
   
    """
    Decoder class used in variational autoencoders (VAEs).

    This class extends the `BiCoder` superclass and applies a neural
    network to map latent representations into reconstructions.
  
    The logsigma_x is fixed and defined upon object creation

    Args:
        neural_net: A Keras model that outputs a reconstruction
            
    Attributes:
        neural_net: The underlying neural network used to decode inputs.
    """   
        
    
    def __init__(self,neural_net):
        super().__init__(neural_net)
        self.sigma_x = 0.75
        self.logsigma_x = tf.math.log(self.sigma_x)
        
    def call(self,z):
        """
        call() method 
        Respects the contract and implements the call() method
        Returns tuple of the noisy reconstruction xhat, the mean mu_x and logsigma_x

        """
        mu_x = self.neural_net(z)
        eps = tf.random.normal(tf.shape(mu_x))

        xhat = mu_x+eps*self.sigma_x
        
      
        return xhat, mu_x,self.logsigma_x
    

