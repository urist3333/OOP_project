from tensorflow.keras import layers

class BiCoder(layers.Layer):
    """
    Superclass for  Encoder and  Decoder classes.

    Inherits from tensorflow.keras.layers.Layer. 
    Sets the contract that all subclasses must utlize a call() method

    Args:
        neural_net: A Keras neural network model

    Attributes:
        neural_net: The neural network used to 'BiCode' the input.
    """
    def __init__(self, neural_net):
        super().__init__()
        self.neural_net = neural_net

    def call(self, inputs):
        """
        Abstract method call()  
        Enforces that all subclasses of BiCoder must override call() from tensorflow.keras.layers.Layer
        """
        # Abstract method
        raise NotImplementedError(
            "All subclasses of BiCoder must implement a call method"
        )


