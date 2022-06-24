# cvnets

## TODOs

-   [x] Add mixed-based augmentation into training process
-   [ ] Auto-resume training
-   [ ] Explanable AI
-   [ ]

## Usage

### Install

Clone the repository and install environments

```bash
$ git clone https://github.com/jimmylin0979/cvnets
$ cd cvnets
$ pip install -r requirements.txt
```

### Training

All of the training infos, including location of dataset, train/valid split ratio, model and so on are written in config.yaml.  
Please check the config.yaml first to customize your network.

`python main.py --mode train --logdir <folder-name>`

### Evaluating

`python main.py --mode eval --logdir <folder-name>`

## Reference
