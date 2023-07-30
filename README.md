# Introduction
This project trains StyleGAN2-ADA-pytorch on the positive(abnormal) and negative(normal) bone x-rays from the MURA database to create realistic fake images to add to the dataset of the CNN. The purpose of generating this synthetic data is to add a higher volume of bone x-rays, specifically shoulder x-rays, to potentially boost the accuracy of the CNN model.
### Image Comparisons

| Real Images | Synthetic Images | 
| ----------- | ----------- |
| ![](./figures/real-image-grid1.png) | ![](./figures/fake-image-grid1.png) |

# How to Run
To generate synthetic data run the Google Colab notebook [here](https://colab.research.google.com/drive/1VgWEs76z3hTT6XkvFMnM1qS8Upgjq8Tv?usp=sharing)!

To see example synthetic data created by this project click [here!](https://drive.google.com/drive/folders/1PRVGdGPiz3X-xlbXKvaKEm1TSFNPni3N?usp=share_link)

To train a model using this data check out my other repository [aileendugan/bone-xray-cnn](https://github.com/aileendugan/bone-xray-cnn)

# Details
The following has the code for the StyleGAN2-ADA-pytorch training on the positive and negative xrays from the MURA database to create realistic fake images to add to the dataset of the CNN made last semester. There is also a google colab notebook that is my project from last semester but adding in the new extra data with it.
**If the colab notebooks straight from github do not show the code or you need access to any of the data I used or created, the links to them are here:
1. CNN from last semester project edited for implementation of new synthetic data added to training data: https://colab.research.google.com/drive/1l80sIltUi97t2G0qcPGZ_oWnHpXlTS5W?usp=sharing
2. StyleGAN2-ADA-Pytorch implementation for creating the synthetic data: https://colab.research.google.com/drive/1VgWEs76z3hTT6XkvFMnM1qS8Upgjq8Tv?usp=sharing
3. Google drive folder with MURA database: https://drive.google.com/drive/folders/1oi4cwqoqcoI22BcLPptZf0ZoBgLWt-vK?usp=share_link
4. Google drive folder used for making and holding all synthetic data: https://drive.google.com/drive/folders/1PRVGdGPiz3X-xlbXKvaKEm1TSFNPni3N?usp=share_link

# Part 3
**Part 3: First Solution + Validation**
  The architecture I used was StyleGAN2-ADA-Pytorch. I chose to use a StyleGAN because they are pre-trained and the X-ray images from the MURA dataset are complex and don’t have enough samples for the amount of variation thus using styleGAN to create the synthetic images is best as it can learn better. For this, I trained the StyleGAN separately, once for the positive(abnormal) x-ray images, and another time for the negative(normal) x-ray images in order to make sure the labels stayed with the proper fake images. StyleGAN2 was inspired by MSG-GAN but has an architecture to make use of multiple scales of images generation without explicitly requiring the model to do so. This is done using a resnet style skip connection between lower resolution feature maps to the final generated image. This is a big reason I chose to use it in the first place as the MURA database images are not ideal but StyleGAN2 should be able to work with them still. StyleGAN2-ADA also has an adaptive discriminator (ADA=adaptive discriminator augmentation). I also used a pytorch implementation as that seemed to work best in google colab. Below is a photo of the architecture of StyleGAN2 and StyleGAN2-ADA.

![Figure 1. StyleGAN2 and StyleGAN2-ADA architecture](./figures/stylegan-arch.png)
Figure 1. StyleGAN2 and StyleGAN2-ADA architecture

The accuracy achieved can just be shown visually by looking at the fake images at the end of the training that are created and considered to “pass” as real. When looking at these images, they are not all great or passable as real x-rays mainly because some of the negative(normal) x-ray images, you can see curved bones which is not a normal thing. I also picked the shoulder x-rays specifically which I think is a pretty complex bone structure in general so that might have been tough. Some of the fake images look really realistic though. The images can be seen below.

![figure2. negative shoulder x-rays synthetic made from stylegan](./figures/all-neg-shoulder-synthetic.png)
Figure 2. Negative Shoulder x-rays made from StyleGAN2-ADA

![figure3. closeup of negative synthetic bone x-rays](./figures/closeup-sample-neg-synthetic.png)
Figure 3. Close up of the X-rays from figure 2

![figure4. all positive shoulder x-rays synthetic made stylegan](./figures/all-pos-shoulder-synthetic.png)
Figure 4. Positive(abnormal) shoulder X-rays made from StyleGAN2-ADA

![figure5. closeup of positive shoulder synthetic bone x-rays](./figures/closeup-pos-synthetic.png)              
Figure 5. Close up of images from figure 4

From what I observed, it seems that the MURA database x-rays have a lot of variation in general and some of the images the x-ray portion is not fit to the whole image. This could be a big reason as to why the synthetic data made was not that great as more augmentation would have been a good idea like cropping and even sharpening the images from the database first. However, StyleGAN2-ADA should be able to handle this in the images which is why I picked it but still the images generated could be better. The positive images I had to cut the training short after 13 hours (I trained the positive and negative both for around 13 hours) There is a resume training code block for each to be able to train longer and get better images. 

**Part 4: Final solution**
For the final solution, I took the images generated and created copies such that each image in the grids of images was cropped to be its own image, sharpened by a factor of 2, and then saved to a new folder either positive/SHOULDER or negative/SHOULDER to keep track of the positive new images and negative new images within the folder schema

	/content/drive/MyDrive/data/Augmented_Fakes.

Then, each image was added to the pandas training dataframe by their new filename paths and given their corresponding label of "positive" or "negative" as their class in the dataframe. Then the training data frame was turned into the proper ndarrays format for x_train being the images from the filenames in the dataframe and y_train being their corresponding positive/negative class names for their labels. Then each image was resized when the images in the rest of the MURA database were resized as well to fit the proper format for the densenet169 CNN used. Tse synthetic images were added to the training data used to train my CNN from last semester to see if a higher accuracy could be achieved. My assumption was that they would not increase the accuracy just because some of the generated images are not that great. Especially the positive (abnormal) synthetic images as I stopped the StyleGAN2-ADA training for the positive images early in order to finish this project in time. 

In the future, I would of course finish the training for StyleGAN2-ADA in order to get better synthetic images to add to the training data. I also think that in the future I would change from densenet169 architecture to a ResNet architecture for my CNN to see if that performs better. The architecture of the CNN I made and all other information about what I did for densenet169 architecture can be found in my computervision xray project on github. Therefore, I will not re-explain it here but rather provide a link to the github repo that contains all this information(https://github.com/aileendugan/CompVisionBoneXrayProject).

I will, however, explain why I think ResNet would be a good change. ResNet, as opposed to any densenet architecture, adopts a summation of all the proceeding feature maps while densenet concatenates them. The reason that I think this switch to summing the feature maps would be useful is because of all of the unimportant variation in the data. The MURA database, as explained before, has many images where the x-ray is not fit to the entire image and blank space is left. Also, there are many weird things such as some are rotated and some are brighter than others. While this should be fixed with augmenting the data images (another thing to implement/fix in the future), it also seems that summing up the feature maps could help put more importance on the important and consistent features rather than concatenating all of the feature maps and therefore not releasing any unimportant information picked up that could cause more confusion. 

Most of the research papers that I looked at of others using the MURA database did use densenet architectures such as densenet169 like I did or densenet121 which is why I originally chose densenet169. However, without augmenting the data or cleaning it up as much as I should, it seems that ResNet could make a slight difference in a higher accuracy and seems worth trying. Besides switching to ResNet, I could, as mentioned before, try to fix all of the MURA database images through augmentation. The only issue with trying to do this is that not every image in the database needs to be fixed or fixed in the same way and for shoulder x-rays alone there are over 8000 images. Therefore, manually fixing each would be a hassle and I cannot think of a great way to speed up that process. One easy thing to do would be to normalize the coloration, saturation, and hues of each image. I tried last semester to change each x-ray image to grayscale but that did not really do anything but since some of the images are much brighter than others, changing/normalizing the hues across all images could be helpful. This would also likely be helpful for making the synthetic data using the StyleGAN2-ADA.

One of the great things about StyleGAN2-ADA or any StyleGAN is that it is an addition to the GAN architecture that introduces significant modifications to the generator model. GANs or Generative Adversarial Networks have both a generator and a discriminator. The generator takes simple random variables as inputs and generates new data aka the new synthetic images. The discriminator then takes original data and the new generated data and tries to discriminate them, aka tell which one is the synthetic data image, building a classifier. So, the generator tries to learn and train and make more and more realistic fake data to trick the discriminator until the discriminator can no longer correctly pick which data is synthetic and which is real. The ADA addition of StyleGAN2 is an adaptive discriminator augmentation mechanism that significantly stabilizes training in limited data regimes.This makes it helpful for creating synthetic data when training on smaller training datasets. The approach does not require changes to loss functions or network architectures.


For my accuracy results with adding the synthetic data images to the training data for my CNN, I got a 0.8804 or 88.04% accuracy on the training data and 0.6412 or 64.12% accuracy on the testing. Ofcourse, the accuracy from training to testing goes down as to be expected but the same problem for testing is coming up that came up for my CNN results last semester. As shown in figure 2, the model tends to classify x-rays that are actually positive for abnormalities as negative or normal which is the same issue as before. The accuracy is 64% I think mainly because the model classifies the negative x-ray images properly as being negative but cannot detect positive x-rays well at all even with the additional synthetic data. The results can be seen in figures 1 and 2 below.

![figure1. training accuracy CNN with synthetic data](./figures/training-acc-synthetic.png)
 Figure 1. Training Accuracy of CNN with synthetic data added to training data

 ![figure2. testing accuracy CNN with synthetic data](./figures/testing-acc-synthetic.png)
 Figure 2. Testing Accuracy of CNN with synthetic data added to training data

	When comparing  the training and testing accuracy of the CNN with the additional synthetic data to the training and testing accuracy of the CNN from before without the additional synthetic data, the training accuracy improved from about 83% to 88.04% but the testing accuracy decreased from about 67% accuracy to 64.42% accuracy. What I believe caused this is the fact that I trained the CNN on synthetic images that are not that great or “real” looking and so the training accuracy went up because the model looked at those images and made it work however the testing accuracy went down because the model during training took into consideration things about the synthetic images that would never be found in the real images in the test set so the model actually understood less about negative vs positive x-rays from the synthetic images.

It is important to note that my synthetic data is not great as you can see from the figures as part of the part 3 write up. The negative synthetic images are okay but some of the images have curved bones that are not normal and thus obviously fake and do not work as part of the negative (normal) x-ray images and the positive synthetic images are not great because I had to stop the training for them after 13 hours before it was fully complete in order to move on and finish this project. The positive (abnormal) synthetic x-ray images are all a little bit blurrier than desired and do not have clearly pictured bones even compared to the quality of the negative synthetic x-ray images. This likely was the reason for the accuracy of the CNN model not getting better from last year because these new synthetic images are not up to par with the real images from the database. 

I assume that if I trained the StyleGAN2-ADA to completion for both positive and negative x-ray images that this would not be the case and that the accuracy would get better as long as the final synthetic images that fake out the discriminator are more realistic the way I would expect them to be.  Also, it is important to note that the number of training images for shoulder x-rays in the MURA database is around 8000 and with the addition of 480 positive synthetic x-ray images and 480 negative synthetic x-ray images of shoulders created and added to the training data, there becomes about 9330 training images. It is great that the training data was increased by 960 x-ray images however, it is likely that generating more than this to possibly double the training data instead would be beneficial for increasing the accuracy of the CNN. 



