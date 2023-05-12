The following has the code for the StyleGAN2-ADA-pytorch training on the positive and negative xrays from the MURA database to create realistic fake images to add to the dataset of the CNN made last semester. There is also a google colab notebook that is my project from last semester but adding in the new extra data with it.
Part 3: First Solution + Validation
  The architecture I used was StyleGAN2-ADA-Pytorch. I chose to use a StyleGAN because they are pre-trained and the X-ray images from the MURA dataset are complex and don’t have enough samples for the amount of variation thus using styleGAN to create the synthetic images is best as it can learn better. For this, I trained the StyleGAN separately, once for the positive(abnormal) x-ray images, and another time for the negative(normal) x-ray images in order to make sure the labels stayed with the proper fake images. StyleGAN2 was inspired by MSG-GAN but has an architecture to make use of multiple scales of images generation without explicitly requiring the model to do so. This is done using a resnet style skip connection between lower resolution feature maps to the final generated image. This is a big reason I chose to use it in the first place as the MURA database images are not ideal but StyleGAN2 should be able to work with them still. StyleGAN2-ADA also has an adaptive discriminator (ADA=adaptive discriminator augmentation). I also used a pytorch implementation as that seemed to work best in google colab. Below is a photo of the architecture of StyleGAN2 and StyleGAN2-ADA. (they are attached in the github repo under the names of the figures)

Figure 1. StyleGAN2 and StyleGAN2-ADA architecture

The accuracy achieved can just be shown visually by looking at the fake images at the end of the training that are created and considered to “pass” as real. When looking at these images, they are not all great or passable as real x-rays mainly because some of the negative(normal) x-ray images you can see curved bones which is not a normal thing. I also picked the shoulder x-rays specifically which I think is a pretty complex bone structure in general so that might have been tough. Some of the fake images look really realistic though. The images can be seen below.
   
Figure 2. Negative Shoulder x-rays made from StyleGAN2-ADA

Figure 3. Close up of the C-rays from figure 2


Figure 4. Positive(abnormal) shoulder X-rays made from StyleGAN2-ADA
               
Figure 5. Close up of images from figure 4

From what I observed, it seems that the MURA database x-rays have a lot of variation in general and some of the images the x-ray portion is not fit to the whole image. This could be a big reason as to why the synthetic data made was not that great as more augmentation would have been a good idea like cropping and even sharpening the images from the database first. However, StyleGAN2-ADA should be able to handle this in the images which is why I picked it but still the images generated could be better. The positive images I had to cut the training short after 13 hours (I trained the positive and negative both for around 13 hours) There is a resume training code block for each to be able to train longer and get better images. 

Part 4: Final solution
For the final solution, I took the images generated, and created copies such that each image in the grids of images was cropped to be its own image, then each image was resized to fit the size of the images in the rest of the MURA database, as well as sharpened and then savedwith the corresponding proper label. Then they were added to the training data used to train my CNN from last semester to see if a higher accuracy could be acheived. My assumption was that they would not increase the accuracy just because some of the generated images are not that great. I also think that in the future I would change from densenet169 achritecture to a densenet121 architecture for my CNN as it is possible that too many layers caused some issues as well. 


