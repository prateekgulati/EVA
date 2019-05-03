Assignment 1a Collab Link: [EVA-1A.ipynb](https://colab.research.google.com/drive/1BCpxdDAT0HMdCSf5J6EpaCGP93Gt3lRA)
***

<b>Q1. </b><I>What are Channels and Kernels (according to EVA)?</I>
> A <b>Kernel</b> is a randomly allocated weight matrix (usually 3x3) that is traversed and convolved over the image. One kernel extracts a single feature from the image.  
> The output of this covolution can be considered as a set of <b>Channels</b>. Each channel will have similar features.  
For instance, 1 Greyscale Image (NxN) &rarr; convolution with  64 kernels (3x3) &rarr; 64 channels ((N-2)x(N-2))

***
<b>Q2. </b><I>Why should we only (well mostly) use 3x3 Kernels?</I>
>The major reason for using 3x3 kernel is that any odd size kernel can be formed from a 3x3 kernel. So it gives flexibility to either continue using the same size kernel or use multiple 3x3 kernels to generate a bigger size kernel. Also, the reason of using an odd sizw kernel over an even size kernel is that an even size kernel doesn't give a defined line of symmetry. This rules out the option of using odd size kernels.  
Since 3x3 kernels were preferred by majority, so most of the GPUs these days have been desined in such a way that they accelerate 3x3 convolution.


***
<b>Q3. </b><I>How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199? (show calculations)</I>  

>To reduces the image size by (2), it takes 1 3x3 convolution   
To reduce image size by (199-1), it will take 99 3x3 convolutions  

Input Size|	Convolution	|Output Size|
|:----:|:----:|:----:|
|199|	3x3	|197|
|197|	3x3	|195|
|195|	3x3	|193|
|193|	3x3	|191|
|191|	3x3	|189|
|189|	3x3	|187|
|187|	3x3	|185|
|185|	3x3	|183|
|183|	3x3	|181|
|181|	3x3	|179|
|179|	3x3	|177|
|177|	3x3	|175|
|175|	3x3	|173|
|173|	3x3	|171|
|171|	3x3	|169|
|.	|  .  |.  |
|35	|3x3	|33 |
|33	|3x3	|31 |
|31	|3x3	|29 |
|29	|3x3	|27 |
|27	|3x3	|25 
|25	|3x3	|23 |
|23	|3x3	|21 |
|21	|3x3	|19 |
|19	|3x3	|17 |
|17	|3x3	|15 |
|15	|3x3	|13 |
|13	|3x3	|11 |
|11	|3x3	|9  |
|9	|3x3	|7  |
|7	|3x3	|5  |
|5	|3x3	|3  |
|3	|3x3	|1  |
***
