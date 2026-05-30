# HW 4 - Jonas Gstoettenmayr

## Exercise 1

The same as for the last HW, so I will not comment on it.

## Exercise 2

We are training a U-Net architecture for image segmentation. It uses an EfficientNet-B0 backbones and weights for image Net. For this to work we also normalize the data with the specific imagenet mean and std.

Speaking of the data we augment it with a variety of methods (flipping, rotation, translation....). As for the loss we use a equal weighted mixture of Dice and BCE.

The "most training", steepest increase in performance, once again happens between 10 and 20 epochs, much like the last homework, I believe this is just a concidence but still an interesting one.

## Exercise 3

For the standard model trained in the HW we end up with a respectable mean Dice of 0.86 ± 0.02 and IoU of 75% ± 3%. Quite good for "only" 30 Epochs of training.

The best samples are replicated almost flawlessly, the only *real* difference is the thickness of the lines representing the worms, and some minor artifacts in the original input picture (i.e. some dirt inside of the picture) which the model segments as worms.

The two worst images till have rather good mask which catch all the worms, but they include some unwelcome extras, the dirt of the microscope for one.

What is very nicely visable for both the worst and best images is that the model learns to associate black with worm as the black background has a higher probability of being a worm than the white background, while black on white background (the actuall worms and some dirt) have the highest probability.

![alt text](images/std_worst.png)

## Exercise 4

What the model is not very good at is handling noise, even when introducing some very small noise of only 0.005 the performance allready falls of a clif, falling down to ~20%.

![alt text](images/std_qlt_noise.png)

It is not hard to see why, even introducing a little noise already makes the model consider the background much more (as discussed previously the model already give highish probability of the black background being worms) no with only a litte noise it really starts classifing the background as worms.

![alt text](images/std_samples.png)

My theory is that the way we are passing the images is a bit of the problem, we are handing it images with 0-255 range, meaning that the guassian noise can actually modify the values and with the weights being relu based a small change of e.g 255 to 254 could activate one neuron more leading to it being wrongly classified.

This is a big problem for real world application as (almost) all real-world has at least a bit of noise inside of it.

One way to solve this conundrum is to hardcap the input to the model to be either 0 or 255 and nothing in between, so pure black and white without shades of gray. But that would put the burdon of noise "detection" on the data processing, so it is not perfect.

## Experiment

Another solution is to introduce noise already durring the training, as I was curious to see how much it would help I trained another model with noise and tested it as well.

With the same 30 epochs the model achieves a Dice of 0.87 ± 0.02 and an IoU of 76% ± 3%, performance almost exactly equal to the standard model (the improvements are within random margin of error).

For the worst samples it too has the same worst samples and problems (dirt in the image is classified as worms). But interestingly it no longer has a probability difference between the white and black background, meaning it has a better structural understanding of what a worm is than the standard model, the probability of the background being classified as a worm is also much lower than both background in the previous.

![alt text](images/wn_std_worst.png)

Where a big difference becomes quickly apperent though is when run with noise.

![alt text](images/wn_qlt_noise.png)

I trained it with a 50% chance of an image to have gaussian noise with std between  0.001 and 0.02 (roughly our range).

As can be seen the noise values of the previous model barely effect its performance with the drop being quite small and it even manages to hold out with a very steep increase of noise of std=0.1, a surprise to me. But this proves that simply introducing noise durring training not only does not hamper the performance of the model but makes it resistant to noise!

As can be seen in the samples of noise it no longer leads it to artifact the images, but it does make the model more "conservative" with its guesses as  the model simply does not segment some worms if it is not sure. It does not segment the background (as mentioned previously it is much better at segmenting background as background even with noise in the background) or random noise points (at least at these levels) though.

![alt text](images/wn_std_samples.png)

This model would be far more production ready than the standard one.
