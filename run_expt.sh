#!/bin/bash
set -e
python3 train_mnist.py \
        --batch-size 32 --log-frequence 100 --warmup 2 --epochs 10 \
        --alpha 0.4 --beta 0.2 --gamma 0.4 --delta 0.6 --epsilon 0.4 --zeta 0.6 \
        --energy-file nrf_energy.txt --latency-file nrf_latency.txt --memory-file rpi_memory.txt \
        --tb-log balanced

python3 train_mnist.py \
        --batch-size 32 --log-frequence 100 --warmup 2 --epochs 10 \
        --alpha 0.3 --beta 0.2 --gamma 0.3 --delta 0.6 --epsilon 0.6 --zeta 0.6 \
        --energy-file nrf_energy.txt --latency-file nrf_latency.txt --memory-file rpi_memory.txt \
        --tb-log memory_skewed

python3 train_mnist.py \
        --batch-size 32 --log-frequence 100 --warmup 2 --epochs 10 \
        --alpha 0.6 --beta 0.2 --gamma 0.3 --delta 0.6 --epsilon 0.3 --zeta 0.6 \
        --energy-file nrf_energy.txt --latency-file nrf_latency.txt --memory-file rpi_memory.txt \
        --tb-log latency_skewed

python3 train_cifar10.py \
        --batch-size 32 --log-frequence 100 --warmup 2 --epochs 20 \
        --alpha 0.4 --beta 0.2 --gamma 0.4 --delta 0.6 --epsilon 0.4 --zeta 0.6 \
        --energy-file nrf_energy.txt --latency-file nrf_latency.txt --memory-file rpi_memory.txt \
        --tb-log balanced_c10

python3 train_cifar10.py \
        --batch-size 32 --log-frequence 100 --warmup 2 --epochs 20 \
        --alpha 0.3 --beta 0.2 --gamma 0.3 --delta 0.6 --epsilon 0.6 --zeta 0.6 \
        --energy-file nrf_energy.txt --latency-file nrf_latency.txt --memory-file rpi_memory.txt \
        --tb-log memory_skewed_c10

python3 train_cifar10.py \
        --batch-size 32 --log-frequence 100 --warmup 2 --epochs 20 \
        --alpha 0.6 --beta 0.2 --gamma 0.3 --delta 0.6 --epsilon 0.3 --zeta 0.6 \
        --energy-file nrf_energy.txt --latency-file nrf_latency.txt --memory-file rpi_memory.txt \
        --tb-log latency_skewed_c10

python3 train_cifar10.py \
        --batch-size 32 --log-frequence 100 --warmup 2 --epochs 20 \
        --alpha 0.6 --beta 0.2 --gamma 0.3 --delta 0.6 --epsilon 0.3 --zeta 0.6 \
        --energy-file nrf_energy.txt --latency-file nrf_latency.txt --memory-file rpi_memory.txt \
        --tb-log default_c10