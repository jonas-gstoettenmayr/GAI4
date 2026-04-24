# Homework 2 - Jonas Gstoettenmayr

## Quick stats

### Deep ensemble

**The deep ensemble took:**

- 6 minutes to train
- 4.5 minutes to infer

**It achieved:** 11% detection (with 5% false alarm rate)

![alt text](imgs/deep.png)

It required lables.

### Autoencoder

**The Autoencoder took:**

- 1 minute to train
- 20 seconds to infer

**It achieved:** 77% detection (with 5% false alarm rate)

It needed no lables for this!

![alt text](imgs/ae.png)

## Comparison

This is not a fair competition, the Autoencoder absolutely smashed the deep ensemble in every metric.
It was faster to train, infer and on top of that 7 times better at differentiating the OOD samples. For comparison, the deep ensemble would need a leeway of 26% false alarm rate to get the same detection rate as the Autoencoder.

The Autoencoder lead is further enhanced by the fact that it is unsupervised learning (meaning it requires no labels).

My theory for this big difference is that the ensemble's low epoch count of 5 has not let them converge enough to agree on some of the more difficult samples, as can be seen in the very first graph where they can agree on pretty much half the samples, while the other half has some minor to major disagreements. I would assume that increasing the epoch count would let them converge more to each other's local optima and agree more often, on non OOD samples.

Finally this result has some interesting connotations, while the deepensamble is able to give a good measure of how confident it is in its answers, the autoencoder is just overall better in every way of detecting OOD but unable to actually classify any model, nor give a good confidence score for the image. With the deep ensamble the variation (confidence in its answer) is directly linked to the model output, meanwhile a commbination of Autoencoder and classification model would be able to flag OOD samples and also give confidence on samples (based on the MAE diviation of the Autoencoder) but this confidence would be seperate from the model itself and more closely just based on how much the new samples resembles the training set (though it could be argued that with models simply trying to reproduce their training data anyway the deep ensabmle also does this).

## Extra

What is very nice to see in the autoencoder is how it handles colours, as it was trained on rather dark, brownish ground pictures it always tries to reconstruct them as such and leaves out bright colours such as in the image below.

![alt text](imgs/fids_recon.png)