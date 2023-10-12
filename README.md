# Neural-Style-Transfer-Project using Pytorch
The aim of this project is to implement the first descriped method for Neural Style Transfer for artistic images, 
method proposed by Gatys et al. in this paper (https://arxiv.org/pdf/1508.06576.pdf)

# Idea
Have a cool picture or artwork? What would you say if I told you that it is possible to transfer that unique style to the most boring pic? Neural Networks are some very versatile tools and between their applications in arts lays the so called style transfer. It means exactly what its name says, it takes a style from one particular image and modifyies other images to match that style.

# Resources and general information 
For creating this project I've found very useful the following resources. <br />
[1]Pytorch tutorial (https://pytorch.org/tutorials/advanced/neural_style_tutorial.html) <br />
[2]Github repository (https://github.com/gordicaleksa/pytorch-neural-style-transfer) and corresponding Youtube explanations (https://www.youtube.com/watch?v=S78LQebx6jo&list=PLBoQnSflObcmbfshq9oNs41vODgXG-608) <br />
[3]Tensorflow tutorial (https://www.tensorflow.org/tutorials/generative/style_transfer) <br />
[4]Gatys et al. paper <br />
[5]Pytorch Documentation (https://pytorch.org/docs/stable/index.html)

The code for reconstructing the image from different layers and visualize Gram matrix was taken from [2] and slightly modified(cut off unnecessary parts for this project, modified paths variables, etc).

For processing the images I've also followed the method from [2] because there are not too many choices to do that. Vgg19 is trained on ImageNet database, so my input images 
should have the same dimensions as those from ImageNet.<br />
For defining loss functions(content/style loss, gram matrices, etc.) I've implemented the formulas from [4] with the help of [2], [1], and [5]. Here I extended the code
by introducing L1 norm for content loss, modifying reducing parameter(sum or mean) from pytorch available loss functions<br />
The model used was extracted from pre-trained VGG19 network trained on ImageNet dataset (available in Pytorch) and set the requires_grad=False to freeze it's weights. For this I keept the code from [2].<br />
For optimization part, I've followed the idea from [2] and expanations from [1] and [3], extending the code by my needs(used optimizer, number of iterations for backprop, etc.)

# Project structure
* model dir - contains the VGG19 model(extracted layers)
* utils dir - contains image processing functions(loading, tranforming), gram matrices construction, and running configurations(choices for input images, weights, paths, optimizer, etc)
* data dir - contains input images(content and style) and output images(after style transfer)
* the optimization algorithm is found in transfer_algorithm.py
* the reconstructing part is in reconstruct.py

# Running
Generating one stylized image requires a lot of computation, so if CUDA is available on the device, the GPU will be used. For my hardware, for one final output the mean running time is > 20 minutes with CPU (AMD Ryzen7 4800H) and 7-10 minutes with GPU(Nvidia GeForce GTX 1650 4BG).

# Personal work and results
* A written paper with the state of the art, proposed method and analysis: https://drive.google.com/file/d/1eIZcxtUGf3NKOo1J7G2GVVGF9U6eLXgl/view?usp=sharing
* A video presenting the project and the results I've obtained: https://youtu.be/dK62-PEs3cw?si=kuFlukWr6bAljbED
* Poster: https://drive.google.com/file/d/1vgTmUyp8RmoBzNVf1TCRE94IQ83urIBp/view?usp=sharing
