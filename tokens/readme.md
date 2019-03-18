## Tiger Tokens

This module contains code for computing the generative tiger tokens patch distributions.


### Usage

```bash
$ python tigertokens.py output-file -k=NUM-CLUSTERS -n=NUM-PATCHES-PER-IMAGE --patch-size=9
```
You can get usage information with `python tigertokens.py -h`:
```
$ python tigertokens.py -h
usage: tigertokens.py [-h] [--img-size IMG_SIZE] [-k K] [-n N]
                      [--patch-size PATCH_SIZE] [--no-norm] [--off-contours]
                      Output File

positional arguments:
  Output File           Path to file where the tokens will be saved

optional arguments:
  -h, --help            show this help message and exit
  --img-size IMG_SIZE   Size to resize images to. Default: 320
  -k K                  Number of clusters (and number of tokens). Default:
                        400
  -n N                  Number of patches to sample from each image. Default:
                        300
  --patch-size PATCH_SIZE
                        Size of token patches. Default: 15
  --no-norm             Do not normalize patches before clustering. By
                        default, patches are normalized to have the same
                        brightness range
  --off-contours        Sample patches from anywhere in the images, not just
                        centered on contours.
```
