import os 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.manifold import TSNE
import tensorflow as tf
import numpy as np 
def plot_grid(x_recon, dset,batch_size,epochs,name,save_plot,noisy,learning_rate,version =""):
   
    N = 10; C = 10; figsize = (24.,28)
    if dset == "mnist_bw":
        color_map = "viridis"
        plot_name = dset
        x_recon_images = x_recon.numpy().reshape(-1, 28, 28)
    else:
        color_map = None
        plot_name = f"{dset}_{version}" if version else dset
        x_recon = tf.clip_by_value(255*x_recon, clip_value_min=0, clip_value_max=255).numpy().astype(np.uint8)
        x_recon_images = x_recon.reshape(-1, 28, 28,3)

    if noisy:
        plot_name = f"{'noisy'}_{plot_name}"
   
    
 
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111, nrows_ncols=(N, C), axes_pad=0)
    for ax, im in zip(grid, x_recon_images):
        ax.imshow(im, cmap=color_map)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(wspace=0, hspace=0)

    if save_plot ==True:
        save_dir = "figs" 
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{plot_name}_{name}_learning_rate_{learning_rate}_batch_size_{batch_size}_epochs_{epochs}.png")
        print(f"Saved grid as {plot_name}_{name}_learning_rate_{learning_rate}_batch_size_{batch_size}_epochs_{epochs}.png")

   
    plt.show()





def plot_latent(z,labels,dset,batch_size,epochs,name,save_plot,noisy,learning_rate,version =""):
    if dset == "mnist_bw":
        plot_name = dset
    else:
        plot_name = f"{dset}_{version}" if version else dset
    if noisy:
        plot_name = f"{'noisy'}_{plot_name}"
   
    z_embedded = TSNE(n_components=2,learning_rate=100,init="random",perplexity=5,
    ).fit_transform(z)

    plt.scatter(z_embedded[:, 0], z_embedded[:, 1],c = labels,cmap="tab10",s = 5)
    if save_plot ==True:
        save_dir = "figs" 
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{plot_name}_{name}_learning_rate_{learning_rate}_batch_size_{batch_size}_epochs_{epochs}.png")
        print(f"Saved plot {plot_name}_{name}_learning_rate_{learning_rate}_batch_size_{batch_size}_epochs_{epochs}.png")
    
    plt.show()
 