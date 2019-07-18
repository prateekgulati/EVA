## Receptive Field
Receptive Field Calculation for a CNN architecture from Research Paper: **CNN-based Segmentation of Medical Imaging Data** available [here](https://arxiv.org/pdf/1701.03056.pdf).  
Below are the major formulas needed to compute *receptive field* for an architecture:  
![3](https://i0.wp.com/syncedreview.com/wp-content/uploads/2017/05/32.png?resize=372%2C171&ssl=1)

|Convolution|Input|Receptive Field |Output|Kernal	| Strides	| Jin	| Jout	| 	Padding |
|:---:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|1   | 128| 128	| 3	|3	 | 1  	 | 1	 | 1	| 1|
|2   | 128| 64	| 5 |3	 | 2	   | 1	 | 2	|	1|
|3   | 64	| 64	| 9 |3	   | 1	   | 2	 | 2	|	1|
|4   | 64	| 32	| 13|3	   | 2	   | 2 	 | 4	|	1|
|5   | 32	| 32	| 21|3	   | 1	   | 4 	 | 4	|	1|
|6   | 32	| 16	| 29|3	   | 2	   | 4 	 | 8	|	1|
|7   | 16	| 16	| **45**|3	   | 1	   | 8	 | 8	|	1|
***
