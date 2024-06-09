for alpha in 5.0 0.1; do 
    for fed_alg in fed_avg fed_prox fed_opt fed_tree; do
        taskset -c 1-60 python3 train_test.py --alpha $alpha --hid_size 128 --sample_rate 0.1 --learning_rate 0.05  --global_learning_rate 0.004 --num_epochs 200 --en_att --att_name fuse_base --fed_alg $fed_alg --mu 0.01
    done
done
