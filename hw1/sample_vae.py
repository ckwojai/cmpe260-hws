from argparse import ArgumentParser

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from MyVAE import MyVAE


img_dim = (64, 64)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps')


def my_vae_generate(filepath):
    vae = MyVAE(in_channels=3, latent_dim=10).to(device)
    vae.load_state_dict(torch.load(filepath))

    def generate(_):
        out_img = vae.sample(device=device)
        im = out_img[0].detach().cpu().numpy()
        im = im.reshape((3, *img_dim)).swapaxes(0, 2)
        im = (im * 255.0).astype(np.uint8)
        plt.subplot(2, 1, 1)
        plt.imshow(im)
        plt.show()

    plt.subplot(2, 1, 2)
    plt.axis('off')
    b = Button(plt.axes([0.35, 0.1, 0.30, 0.10]), 'Generate')
    b.on_clicked(generate)
    generate(None)
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", type=str, default="vae.pth", help="file name")
    args = parser.parse_args()

    my_vae_generate(args.file)
