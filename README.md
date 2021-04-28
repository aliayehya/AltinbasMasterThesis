**Note : You must download the models folder files from Google Drive which is found in Referencesthe and attach them to the project files**
# Section One: Summary
In this Project, I had used Deep Learning to accurately identify the gender and age of a person from a single image of a face. I used the models Winner of LAP challenge on apparent age estimation,The model was fine-tuned on the dataset of the ChaLearn apparent age estimation challenge. An ensemble of these models led to 1st place at the challenge (115 teams).  For age estimation the output layer has 101 neurons (0-100 years, one for each year). To obtain the predicted age, you need to take the expected value over the softmax-normalized output probabilities. For gender prediction the output layers has 2 neurons (0 for female, 1 for male).


## Section Two: Dataset 
For this project, I had used the IMDB-WIKI dataset, the dataset is available in the public domain . This dataset is the largest publicly available dataset for facial images with gender and age classifications for training, since publicly available facial image datasets are often small to medium-sized, rarely exceeding tens of thousands of photos, often without age information. We decided to collect a large data set. Of celebrities. For this purpose, we took a list of the 100,000 most popular actors as listed on the IMDb site and it crawled (automatically) from their profiles their birthdate, name, gender and all pictures related to this person. Additionally, we crawled all profile pictures from Wikipedia People Pages with the same descriptive information. We removed the photos without a time stamp (the date the photo was taken). Assuming that photos with individual faces likely show the actor and that the timestamp and date of birth are correct, we were able to assign the (real) biological age to each image. Of course, we cannot guarantee the accuracy of custom age information. Besides wrong timestamps, there are many still images from movies - films that can lengthen production times. In total, we got 460,723 face photos from 20,284 celebrities from IMDb and 62,328 from Wikipedia, thus 523,051 in total. The models I used had been trained on this dataset.

Since some photos (especially from IMDb) contain many people, we only use photos where the second strongest face detection is below the threshold. For the network to be equally discriminatory for all ages, we equate the age distribution of the training.





## Section Three: Additional Python Libraries Required 

    OpenCV:
       pip install opencv-python
    dlib:
       pip install dlib






## Section Four: contents of this Project 

* models folder

 *  age folder :
 1.	age.prototxt
 2.	dex_chalearn_iccv2015.caffemodel

* gender folder :
1.	gender.caffemodel
2.	gender.prototxt

* cnn.py

* a few pictures to try the project on

For age and gender, the .prototxt files describe the network configuration and the .caffemodel file defines the internal states of the parameters of the layers.


## Section Five: Usage 
* Download my Repository and models from google drive
* Open your Command Prompt or Terminal and change directory to the folder where all the files are present.
* AltinbasMasterThesis Use Command :

  		python cnn.py  <image_name>
Note: The Image should be present in same folder where all the files are present

## Section Six: Working
 
![Image](https://raw.githubusercontent.com/aliayehya/AltinbasMasterThesis/main/test2result.PNG) 
 ## Section Seven: References
* data sets:
[Link](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
* Dlib frontal face detector:
[Link](http://dlib.net/face_detector.py.html)
* Models & project folder :
[Link](https://drive.google.com/drive/folders/1ytyiapFi5JRM8tfhz3h_k_LzGJwXt58M?usp=sharing)
* Drawing functions: [Link](https://docs.opencv.org/master/dc/da5/tutorial_py_drawing_functions.html) 
