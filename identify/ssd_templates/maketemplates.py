import torch
import numpy as np
import argparse


def create_templates(files, dest, n):
    '''
    '''
    if type(files) == str:
        files = [files]
    for path in files:
        toks = np.load(path)
        k = int(np.sqrt(toks.shape[-1]))
        toks = toks[np.random.choice(toks.shape[0], n, replace=False)]
        toks = torch.from_numpy(toks).view(-1, k, k).unsqueeze_(1).float()
        toks = toks * 2 - 1
        torch.save(toks, dest + f'templates{n}-{k}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+',
        help='Path to .npy file containing tokens')
    parser.add_argument('-d', '--dest', default='./',
        help='Path to destination directory where output will be saved.')
    parser.add_argument('-n', '--num-tokens', type=int,
        help='Number of tokens to use in creating templates. A template is '
             'created for each token used.')
    args = parser.parse_args()

    create_templates(args.files, args.dest, args.num_tokens)


if __name__ == '__main__':
    main()

