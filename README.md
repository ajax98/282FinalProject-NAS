```shell
python train_cifar10.py --batch-size 32 --log-frequence 100 --warmup 2 --epochs 90 --alpha 0.2 --beta -0.2 --gamma 0.6 --delta 0.6 --energy-file rpi_energy.txt --latency-file rpi_speed.txt --tb-log model_name
```

### Train CIFAR10 over webpage
You could also run the program using the webpage implementation provided in this repo. 
Follow these steps:
```shell
npm install 
node app.js
```=
**Note:** I use + not * in loss.


# 282FinalProject-NAS
An attempt at using Neural Architecture Search for constrained systems

