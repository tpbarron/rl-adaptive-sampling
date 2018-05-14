
for i in 1 2 3 4 5
do
    python ars.py --env_name CartPoleContinuous-v0 -nd 1 -du 1 -s 0.05 -std 0.05 --n_iter 2000 --dir_path exps/cartpole_d1/${i} --seed ${i}
    python ars.py --env_name CartPoleContinuous-v0 -nd 2 -du 2 -s 0.05 -std 0.05 --n_iter 2000 --dir_path exps/cartpole_d2/${i} --seed ${i}
    python ars.py --env_name CartPoleContinuous-v0 -nd 5 -du 5 -s 0.05 -std 0.05 --n_iter 2000 --dir_path exps/cartpole_d5/${i} --seed ${i}
    python ars.py --env_name CartPoleContinuous-v0 -nd 10 -du 10 -s 0.05 -std 0.05 --n_iter 2000 --dir_path exps/cartpole_d10/${i} --seed ${i}
done
