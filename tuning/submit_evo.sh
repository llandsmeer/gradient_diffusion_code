set -ex

for i in `seq 0 1000`
do
python3 ./c4a_opt_cmaes.py $i $1
done
