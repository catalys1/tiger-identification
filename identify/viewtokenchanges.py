import torch
import argparse
import matplotlib.pyplot as plt


def diff_image(tokens, model):
    tok = torch.load(tokens)
    w = torch.load(model, map_location='cpu')
    w = w['ssd.weight']
    diff = torch.abs(w - tok).squeeze_().view(8, 8, 19, 19).permute(0,2,1,3)
    diff = diff.contiguous().view(8 * 19, 8 * 19)

    return diff


def display(img):
    plt.imshow(img)
    plt.show()


def main():
    parser = argparse.ArgumentParser('See how tokens changed during training')
    parser.add_argument('model', help='Path to saved model parameters')
    parser.add_argument('-t', default='templates', help='Initial tokens file')
    args = parser.parse_args()

    diff = diff_image(args.t, args.model)
    display(diff)


if __name__ == '__main__':
    main()

