## Usage

 - To run VAE with RealNVP prior on CelebA, please execute:
```
python main.py --model VAE --network densenet64 --prior RealNVP
```

 - Otherwise, to run srVAE:
```
python main.py --model srVAE --network densenet32x64 --prior RealNVP
```

#### *Credits*
This work was derived from the implementation of Ioannis Gatopoulos, 2020 https://github.com/ioangatop/srVAE