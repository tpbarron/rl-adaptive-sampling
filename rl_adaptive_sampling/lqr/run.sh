

declare -a positions=(
                      "--x0 0.5 --y0 0.5"
                      "--x0 0.5 --y0 0.5 --xv0 0.1 --yv0 -0.1"
                      "--x0 -0.5 --y0 -0.5 --xv0 0.1 --yv0 -0.1"
                      "--x0 -0.5 --y0 -0.5"
                      "--x0 0.5 --y0 -0.5 --xv0 0.1 --yv0 -0.1"
                      "--x0 0.5 --y0 -0.5"
                      "--x0 -0.5 --y0 0.5 --xv0 0.1 --yv0 -0.1"
                      "--x0 -0.5 --y0 0.5"
                      )

BASE_DIR="/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/rl_adaptive_sampling/lqr/";

for pos in "${positions[@]}"
do
    for i in 1 2 3 4 5 6 7 8 9 10
    do
        # run all variants

        # kf versions use large batch just as placeholder
        python pth_pg.py ${pos} --batch-size 10000 --lr 0.005 --use-diagonal-approx --kf-error-thresh 0.1 --max-samples 500000 --log-dir ${BASE_DIR}/data/variants1/ --seed ${i} &
        python pth_pg.py ${pos} --batch-size 10000 --lr 0.005 --use-diagonal-approx --kf-error-thresh 0.2 --max-samples 500000 --log-dir ${BASE_DIR}/data/variants1/ --seed ${i} &
        python pth_pg.py ${pos} --batch-size 10000 --lr 0.005 --use-diagonal-approx --kf-error-thresh 0.3 --max-samples 500000 --log-dir ${BASE_DIR}/data/variants1/ --seed ${i} &
        python pth_pg.py ${pos} --batch-size 10000 --lr 0.005 --use-diagonal-approx --kf-error-thresh 0.4 --max-samples 500000 --log-dir ${BASE_DIR}/data/variants1/ --seed ${i} &
        python pth_pg.py ${pos} --batch-size 10000 --lr 0.005 --use-diagonal-approx --kf-error-thresh 0.5 --max-samples 500000 --log-dir ${BASE_DIR}/data/variants1/ --seed ${i} &

        wait;

        # kf versions use large batch just as placeholder
        python pth_pg.py ${pos} --batch-size 10000 --lr 0.005 --kf-error-thresh 0.1 --max-samples 500000 --log-dir ${BASE_DIR}/data/variants1/ --seed ${i} &
        python pth_pg.py ${pos} --batch-size 10000 --lr 0.005 --kf-error-thresh 0.2 --max-samples 500000 --log-dir ${BASE_DIR}/data/variants1/ --seed ${i} &
        python pth_pg.py ${pos} --batch-size 10000 --lr 0.005 --kf-error-thresh 0.3 --max-samples 500000 --log-dir ${BASE_DIR}/data/variants1/ --seed ${i} &
        python pth_pg.py ${pos} --batch-size 10000 --lr 0.005 --kf-error-thresh 0.4 --max-samples 500000 --log-dir ${BASE_DIR}/data/variants1/ --seed ${i} &
        python pth_pg.py ${pos} --batch-size 10000 --lr 0.005 --kf-error-thresh 0.5 --max-samples 500000 --log-dir ${BASE_DIR}/data/variants1/ --seed ${i} &

        wait;

        # regular pg version only change batch size
        python pth_pg.py ${pos} --batch-size 100 --lr 0.005 --max-samples 500000 --log-dir ${BASE_DIR}/data/variants1/ --no-kalman --seed ${i} &
        python pth_pg.py ${pos} --batch-size 500 --lr 0.005 --max-samples 500000 --log-dir ${BASE_DIR}/data/variants1/ --no-kalman --seed ${i} &
        python pth_pg.py ${pos} --batch-size 1000 --lr 0.005 --max-samples 500000 --log-dir ${BASE_DIR}/data/variants1/ --no-kalman --seed ${i} &
        python pth_pg.py ${pos} --batch-size 5000 --lr 0.005 --max-samples 500000 --log-dir ${BASE_DIR}/data/variants1/ --no-kalman --seed ${i} &

        wait;
    done
done
