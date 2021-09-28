from __future__ import print_function

import torch
import torch.nn
from utils.parse import ArgumentParser
import utils.extract_opts as opts
import matplotlib.pyplot as plt
from torch.autograd import Variable
from model import HyperspecAE
from train_utils import get_dataloader

def extract_abundances(opt):

    _, test_set = get_dataloader(BATCH_SIZE=opt.batch_size, DIR=opt.src_dir)
    model = HyperspecAE(opt.num_bands, opt.end_members, opt.gaussian_dropout, opt.activation,
                opt.threshold, opt.encoder_type)
    
    
    checkpoint = torch.load(opt.ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to('cpu')

    N_COLS = 3
    N_ROWS = 1
    view_data = [test_set[i][0] for i in range(N_ROWS * N_COLS)]
    plt.figure(figsize=(25, 25))
    model.eval()
    with torch.no_grad():
        for i in range(N_ROWS * N_COLS):
            # original image
            r = i // N_COLS
            c = i % N_COLS + 1

            # reconstructed image
            ax = plt.subplot(2 * N_ROWS, N_COLS, 2 * r * N_COLS + c)
            img = test_set.train_data
            img = torch.tensor(img)
            e, y = model(img.float())
            plt.imshow(e.detach().squeeze().numpy().T[i].reshape(95, 95))
            #plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.savefig(f'{opt.save_dir}/abundances.png')
    plt.show()
    
def extract_endmembers(opt):

    _, test_set = get_dataloader(BATCH_SIZE=opt.batch_size, DIR=opt.src_dir)
    model = HyperspecAE(opt.num_bands, opt.end_members, opt.gaussian_dropout, opt.activation,
                opt.threshold, opt.encoder_type)


    checkpoint = torch.load(opt.ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to('cpu')
    N_COLS = 3
    N_ROWS = 1

    view_data = [test_set[i][0] for i in range(N_ROWS * N_COLS)]
    plt.figure(figsize=(25, 25))
    model.eval()
    with torch.no_grad():
        for i in range(N_ROWS * N_COLS):
            # original image
            r = i // N_COLS
            c = i % N_COLS + 1

            # reconstructed image
            ax = plt.subplot(2 * N_ROWS, N_COLS, 2 * r * N_COLS + c)
      
            e = model.decoder.weight
            plt.plot(e.detach().squeeze().numpy().T[i])
    plt.savefig(f'{opt.save_dir}/end_members.png')
    plt.show()
    
def _get_parser():
    parser = ArgumentParser(description='extract.py')

    opts.model_opts(parser)
    opts.extract_opts(parser)
    
    return parser
    
def main():
    parser = _get_parser()

    opt = parser.parse_args()
    extract_abundances(opt)
    extract_endmembers(opt)


if __name__ == "__main__":
    main()