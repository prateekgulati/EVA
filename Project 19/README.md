


## Object Detection using tinyYOLO v3
 ### Goal
 Build a object detection model for a custom unseen object using a pretrained YOLO v3 on Google Colab

### Data PreProcessing

 - Data Collection: Gather around 100 JPG files for the object and store in a folder
- Image Annotation: Annotate each image by creating a bounding box around the required object
- Data Preprocessing: Combine the image with its annotated text file and put them together in a folder. Split the data in test and train
- Anchor Box: Calculate the anchor box sizes using clustering. 3 different anchor boxes are used in this experiment.

Aggregate all this data in a single folder and upload on colab. 
### Further steps

 - Run the EVA_19.ipynb file on google colab.
- Download all the required packages
- Clone [AlexeyAB's darknet](https://github.com/AlexeyAB/darknet/) for the YOLOv3 model
- Load the pretrained model weights from [Joseph's repo](https://pjreddie.com/media/files/yolov3.weights)
- Validate the results of the pretrained model
- Train the model on custom object
- Test the object by detecting it in a video
### Results  
#### Pretrained Model validation  

![Pretrained model](Project%2019/assets/YoloPretrainedPrediction.png)
  
#### Training the model on custom object  

Custom object : Morty  

<img src="Project%2019/assets/YoloMorty1.png" width=750px align="centre"/>
<img src="Project%2019/assets/YoloMorty2.png" width=750px align="centre"/>

#### GIF  

<img src="Project%2019/assets/SchwiftyGif.gif" width=750px align="centre"/>

#### Youtube Video:  

[![Watch the video](https://img.youtube.com/vi/-3Ki16nlOS0/hqdefault.jpg)](https://www.youtube.com/watch?v=-3Ki16nlOS0)



### Reference: 
[Train your own yolo v3](https://medium.com/@today.rafi/train-your-own-tiny-yolo-v3-on-google-colaboratory-with-the-custom-dataset-2e35db02bf8f)
