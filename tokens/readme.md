## Tiger Tokens

This module contains the code for computing the "tiger tokens". This includes sampling and clustering patches. The important code is in `tigertokens.py`.

Run `python tigertokens.py -h` to see the commandline arguments. You can specify things such as patch size and number of clusters at run time.

`make-tokens.sh` is a simple bash script that will run the program multiple times for different sized patches. This could be extended to other hyperparameters.