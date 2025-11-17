from nn import encoder_mlp, decoder_mlp, encoder_conv, decoder_conv

def network_selecter(dset):
    """
    network_selecter function
    Selects the appropriate encoder and decoder based on the dataset.
    Args:
        dset: mnist_bw, mnist_color
    Returns:
    encoder_conv, decoder_conv for dset = mnist_color
    encoder_mlp, decoder_mlp for dset = mnist_bw    
            
    """

    if dset == "mnist_color":
        return encoder_conv,decoder_conv
    else:
        return encoder_mlp,decoder_mlp

    