python3 train_cifar10.py \
        --batch-size 32 --log-frequence 100 --warmup 2 --epochs 90 \
        --alpha 0.2 --beta 0.2 --gamma 0.6 --delta 0.6 \
        --energy-file rpi_energy.txt --latency-file rpi_power.txt \
        --tb-log test_cifar10


