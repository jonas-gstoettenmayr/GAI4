# Homework 1 - Jonas Gstöttenmayr

## Exercise 1

Using the provided notebook doing the provided tasks was not difficult.

The data consists of 21 classes spread among 2100 images with 100 images per class.
The test/val/train split is 70/15/15 per class leading to 1470 training and 315 test and validation each.

Lastly when printing out the number of samples I printed them in 3 rows of 7 instead of a long list for better readability.

Each image is 256x256 pixels in size

## Exercise 2 & 3

In this EX we use the previous training data to actually train a model, we use the pretrained CNN efficientNet-B0.

We load the EfficientNet-0  Architecture and the images using TIMM, a pytorch package for images.

In the first part we simply train the architecture of the EfficientNet and try to train it up from the ground using our data, which works out not so well within 20 epochs. Still much better than i expected though.

In the second part we fine-tune the already trained EfficientNet which allows it to simply adjust to our data instead of having to learn it all new from scratch, this leads as one would expect to far better accuracy far quicker.

An interesting observation here is how the loss and accuracy development differ between the pretrained and newly trained EfficientNet. The newly-trained one has a linear increase in accuracy while the pretrained one has a logistic growth where the early epochs lead to a great increase in accuracy before it tapers out and the model has adjusted to its new task. In this phase the improvements are minimal as the model needs to actually adjust how it analyses the pictures to the new dataset instead of simply adjusting how it classifies the images from the pretrained CNN layers.

For the scratch model we had a final accuracy of ~34% on the test set (the classes are balanced so the balanced accuracy is the same). While the pretrained model has a astonishing accuracy of over 96%!

## Exercise 4

### Accuracy curves

When plotting the loss and accuracy curve side by side an interesting observation is the volatility of the from scratch model when tested on validation data, while the training smoothly linearly increases accuracy and decreases loss, on the validation side it is far more volatile fluctuating between being better and worse than the last epoch constantly (although a trend is still visible).

Less interesting, because of how expected it is, but also more visible is the sheer performance difference between the scratch and pretrained model, with the pretrained one roughly already starting at the same accuracy and loss as the from scratch model achieves after 20 epochs, but from that point on the development is logistic, unlike the linear one from the scratch model, leading to significant initial accuracy gains for the pretrained model before it can no longer improve much more after only a couple epochs (the steepest increase in accuracy happens within the first five epochs).

### Confusion Matrices

When looking at the data from a human perspective I personally would have the most difficulty differentiating intersection, medium residential, mobilehomepark and dense residential as they look highly similar (roads with houses). After that I think agricultural and sparse residential might also be hard to differentiate as they often come in one package and one picture could include both classes, the same could be said for buildings and storage tanks.

The scratch model seems to share my opinion somewhat as it was unable to categorize any mobilehomepark, or sparse residential correctly. Though not in what they look similar in as most mobile homeparks were classified as parkinglots and most sparse residentials as dense residentials. Here it comes back to my prediction though as the scratch model seems to have difficulty differentiating the residentials, misclassifying the majority of medium residentials as dense residentials, the same for buildings.

The pretrained model has done much better and seems to have learned the difference between the residential areas as of its 10 misclassifications it only got up to two wrong per category. Two categories that the pretrained shares with the scratch model in finding it a bit difficult are buildings and sparse residential categories (though of course to a much, much lesser degree).
