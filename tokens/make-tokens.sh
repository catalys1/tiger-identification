
SIZES="9 11 13 15 17 19 21 23 25"
N=400

for s in $SIZES
do
    python tigertokens.py tokens$N-$s --img-size=320 --patch-size=$s --norm
done

