
for i in 1 2 3 4 5
do
    python ars.py --env_name CartPoleContinuous-v0 -nd 1 -du 1 -s 0.05 -std 0.05 --n_iter 2000 --use_kf --kf_error_thresh 0.1 --kf_sos 0.0 --dir_path exps/cartpole_e0.1/${i} --seed ${i}
    python ars.py --env_name CartPoleContinuous-v0 -nd 1 -du 1 -s 0.05 -std 0.05 --n_iter 2000 --use_kf --kf_error_thresh 0.2 --kf_sos 0.0 --dir_path exps/cartpole_e0.2/${i} --seed ${i}
    python ars.py --env_name CartPoleContinuous-v0 -nd 1 -du 1 -s 0.05 -std 0.05 --n_iter 2000 --use_kf --kf_error_thresh 0.3 --kf_sos 0.0 --dir_path exps/cartpole_e0.3/${i} --seed ${i}
    python ars.py --env_name CartPoleContinuous-v0 -nd 1 -du 1 -s 0.05 -std 0.05 --n_iter 2000 --use_kf --kf_error_thresh 0.4 --kf_sos 0.0 --dir_path exps/cartpole_e0.4/${i} --seed ${i}
    python ars.py --env_name CartPoleContinuous-v0 -nd 1 -du 1 -s 0.05 -std 0.05 --n_iter 2000 --use_kf --kf_error_thresh 0.5 --kf_sos 0.0 --dir_path exps/cartpole_e0.5/${i} --seed ${i}
done

#python ars.py --env_name Hopper-v1 -nd 60 -du 60 -s 0.02 -std 0.02 --n_iter 8000 --use_kf --kf_error_thresh 0.1 --kf_sos 0.0 --shift 1 --dir_path exps/hopper1 &
#python ars.py --env_name Hopper-v1 -nd 60 -du 60 -s 0.02 -std 0.02 --n_iter 8000 --use_kf --kf_error_thresh 0.05 --kf_sos 0.0 --shift 1 --dir_path exps/hopper2 &
#python ars.py --env_name Hopper-v1 -nd 60 -du 60 -s 0.02 -std 0.02 --n_iter 8000 --use_kf --kf_error_thresh 0.01 --kf_sos 0.0 --shift 1 --dir_path exps/hopper3 &
#wait;

#python ars.py --env_name HalfCheetah-v1 -nd 60 -du 60 -s 0.02 -std 0.03 --n_iter 15000 --use_kf --kf_error_thresh 0.1 --kf_sos 0.0 --dir_path exps/cheetah1 &
#python ars.py --env_name HalfCheetah-v1 -nd 60 -du 60 -s 0.02 -std 0.03 --n_iter 15000 --use_kf --kf_error_thresh 0.05 --kf_sos 0.0 --dir_path exps/cheetah2 &
#python ars.py --env_name HalfCheetah-v1 -nd 60 -du 60 -s 0.02 -std 0.03 --n_iter 15000 --use_kf --kf_error_thresh 0.01 --kf_sos 0.0 --dir_path exps/cheetah3 &
