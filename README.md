# Are_Neural_Nets_Dirty
In this repo we explore a neural net trained on NSFW and SFW images as a binary classifier and then we explore if neural net classifies illusions as NSFW or SFW.

## Dataset
We crawl Reddit using PRAW (https://praw.readthedocs.io/en/latest/) which is Python Reddit API. We collect images from different subreddits and store them in NSFW and SFW folders. 
The dataset is converted to numpy arrays for easier reuse. I scrapped 500 NSFW and 500 SFW images from Reddit. The numpy arrays can be download from this link: https://drive.google.com/open?id=1r0OglnNddeVSjFDDmFZOvvHdE5yzO5gE.
To train, you can load the data as: 
```
  images = np.load('images_224.npy')
  labels = np.load('labels_224.npy')
```

## Training
The dataset is prepared and fed into the neural network. We use AlexNet as a part of the initial experiment. The original AlexNet is used as it is and `alexnet.py` is taken from the official tensorflow repository (https://github.com/tensorflow/models/blob/master/research/slim/nets/alexnet.py).
To train the neural network, type `python3 train.py`.

## Testing
For testing, either refer to the Python notebook or use the `inference.py` code. 
