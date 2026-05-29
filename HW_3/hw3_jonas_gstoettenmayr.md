# HW 3 - Jonas Gstöttenmayr

## Exercise 1

The dataset consists of 100 unique images, with each unique image having 2 variants:

- a mask with where only the worms are visible (the GAN input)
- the actuall image (the target to generate)

The preprocessing consists of multiple steps:

- actually getting the data as images by extracting it from the downloaded format
- converting the image from 16 to 8 bit (saves VRAM)
- resizing the images to 512x512 so that all have a uniform size to input into the model
- converting the mask into a RGB image with values of a range up to 255 instead of 0-1, grayscale

## Exercise 2

I did as instructed but upped the batch size to 4 (to increase the training speed). My first try was with a batch size of 8 but sadly this leads to OOM.

![alt text](images/nvidea_usage.png)

```batch
azureuser@ml-ais-s2410929009:~/localfiles/jonas_g/GAI4/HW_3$ ls -lthr pix2pixHD/checkpoints/bbbc010_512/ | grep pth
-rw-r--r-- 1 azureuser azureuser 696M May 29 20:23 5_net_G.pth
-rw-r--r-- 1 azureuser azureuser  22M May 29 20:23 5_net_D.pth
-rw-r--r-- 1 azureuser azureuser 696M May 29 20:27 10_net_G.pth
-rw-r--r-- 1 azureuser azureuser  22M May 29 20:27 10_net_D.pth
-rw-r--r-- 1 azureuser azureuser 696M May 29 20:35 20_net_G.pth
-rw-r--r-- 1 azureuser azureuser  22M May 29 20:35 20_net_D.pth
-rw-r--r-- 1 azureuser azureuser 696M May 29 20:49 latest_net_G.pth
-rw-r--r-- 1 azureuser azureuser  22M May 29 20:49 latest_net_D.pth
-rw-r--r-- 1 azureuser azureuser 696M May 29 20:49 40_net_G.pth
-rw-r--r-- 1 azureuser azureuser  22M May 29 20:49 40_net_D.pth
```

## Exercise 3

To answer the questions of the notebook:

### Does SSIM plateau

Eventually yes (a very nice s curve is currently visible in the training steps). But within our 40 epochs it has not yet plateaued and would most likly not for a quite a few more epochs as the model irons out some of its bumps. (It has "only" achieved 80% accuracy yet)

**SIMM Summary:** The SIMM steadily improves, meaning the model gets better, over the epochs, the highest rate of growth was beteen 10 and 20 epochs with it beeing slower before and slowing down after again, but as mentioned more epochs would most likly allow it to grow even better (especially with what it could still improve as listed below). The Spread over the SIMM between images seems to tighten over the epochs with the first having +- ~5 % and the 40th epoch only having +- ~3% other than one outlier with only a deviation of -6 % from the mean.

![alt text](images/simm.png)

### Do you see any artifacts in generated samples

For fairness to the GAN here I used only the last epoch (40). In the above image (enlarged to better see the images), yes you can see that the GANs generation is far from perfect.

- The worms are all mono colour and have none of the transperancy and colour gradient of the original
- The sides of the "dish/image?" - the white thing of the worms - has some artifacting with non-smooth edes and light colour creeping, making it seem more blurry
- the image is more blurry overall
- The colour of the background is slightly off

### Random Sample

In my opnion  the quickest and easiest way to optain a new not before seen mask was to just draw one, it is also more fun than simply generating a random tensor of the correct size.

In it i tried to draw a variety of shapes to see how it interacts with the GAN and the answer is not too well, while the image becomes a lot sharper over the epochs it does not make it quite perfect.

In my sample it simply painted the thick line in black instead of making it look like a worm at all, the smaller ones are at least examples. It also did not like the informatin on the sides (as most of the worms are in a certain radius around the center it focuses much more on that) and almost covered them completely in black.

![alt text](images/self_sample1.png)
![alt text](images/self_sample2.png)