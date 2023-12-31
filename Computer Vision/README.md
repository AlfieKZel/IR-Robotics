# Gesture Recognition to Control a Robotic Arm

## Project Overview 

This project was developed as an undergraduate dissertation for the Instrumentation, Automation and Robotics Engineering course at the Federal University of ABC (UFABC).

The goal of this project is to control entirely by computer vision a robotic arm developed using 3D printing and electronics with Arduino as a microcontroller, controlling the joints of the robotic arm by capturing dynamic images of both hands of an operator. 

![tg](https://user-images.githubusercontent.com/21988243/216863371-609cd738-44ab-485f-bbc1-a5000a6fc4ab.gif)

The left hand is responsible for indicating which joint will be moved by through a preconfigured gesture while the right hand is responsible for controlling joint expansion and contraction.

The joint to be controlled is selected according to the gestures below:

<div align="center">
<table>
  <thead>
    <tr>
      <th>Gesture</th>
      <th>Joint</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><img src="https://user-images.githubusercontent.com/21988243/216859724-1a280361-cd66-4fb7-bfc4-12f60466469d.png" width=60%></td>
      <td align="center">Wrist</td>
    </tr>
    <tr>
      <td align="center"><img src="https://user-images.githubusercontent.com/21988243/216859765-6505d3da-10bd-4140-96a3-855d32de1e1c.png" style="width:60%"></td>
      <td align="center">Thumb</td>
    </tr>
    <tr>
      <td align="center"><img src="https://user-images.githubusercontent.com/21988243/216859796-c2c76a18-52bb-413f-b7d9-3108460b4a61.png" style="width:60%"></td>
      <td align="center">Index</td>
    </tr>
    <tr>
      <td align="center"><img src="https://user-images.githubusercontent.com/21988243/216859809-26d539a8-4dc7-4fb7-bea6-2f4e2e4953da.png" style="width:60%"></td>
      <td align="center">Middle</td>
    </tr>
    <tr>
      <td align="center"><img src="https://user-images.githubusercontent.com/21988243/216859818-c12ebcb0-889c-41b5-aa30-126806a4085a.png" style="width:60%"></td>
      <td align="center">Ring</td>
    </tr>
    <tr>
      <td align="center"><img src="https://user-images.githubusercontent.com/21988243/216859822-6c409318-c611-495a-bac4-5eaaffbca52a.png" style="width:60%"></td>
      <td align="center">Pinky</td>
    </tr>
  </tbody>
</table>
</div>

The expansion and contraction of the joint selected by the left hand is controlled by the distance between the thumb and index finger of the right hand, being that the greater the distance between these two fingers, the greater will be the expansion of the joint until
maximum expansion or maximum contraction occurs, as shown below:

![diversos-graus](https://user-images.githubusercontent.com/21988243/216860314-194199ea-ea06-4f00-9d69-bf5e62d8cf5e.png)

**P.S.**: the project was done in group, the other members of the group were responsible for developing the robotics arm with 3D printing and building its electronic with Arduino as microcontroller while **I was responsible for developing the entire code of this repository, doing the computer vision script and its interaction with the Arduino**. For information about the construction of the robotic arm and its design please refer to the [monography.pdf](https://github.com/Brunocds/computer-vision-robot-control/blob/main/reports/monography.pdf) file in the reports folder.

## Architecture

The computer vision of the project has the following architecture:

![general-architecture](https://user-images.githubusercontent.com/21988243/216868930-633dfbce-9c69-4331-ab5c-cbc0ba73785a.png)

* **Image capture and processing:** using OpenCV through Python is possible to capture an image from the webcam as a vector or matrix of pixels, each pixel containing information about the values of the red, green and blue colors (from 0 to 255 for each of them) using the RGB system. 
* **Hand recognition:** the matrix of pixels in the previous step is used as input to MediaPipe, a ready-to-use cutting-edge ML solution framework that has in one of its solutions the hand recognition. The output of the MediaPipe is the coordinates x,y and z of 21 joints of the hands identified in the image.
* **Gesture recognition:** a MLP (Multilayer Perceptron) artificial neural network model is trained using the joints coordinates captured by MediaPipe and uses it as input to detect what hand gesture was done.
* **Distance ratio between finger tips:** the coordinates captured by MediaPipe are used to calculate the distance ratio between the thumb and index tips through Python using Numpy and simple geometry. 

## Getting started (on Windows)

1. Clone the source code by running:
```bash
$ git clone https://github.com/Brunocds/computer-vision-robot-control.git
```
2. Navigate to the project root folder by running:
```bash
$ cd computer-vision-robot-control
```
3. Setup a virtual environment:
```bash
$ python -m venv venv 
$ source venv/Scripts/activate
$ pip install -r requirements.txt
```
4. Run the application:
```bash
$ python apply-model.py 
```
When running the application you can specify two options:
* --device: the camera device number used by OpenCV. Usually is 0, but if the code don't run you can try other, e.g.:
 ```bash
 $ python apply-model.py --device 0
 ```
* --arduino_mode: if you'd like to run only the computer vision side of the application use any parameter different than 1 (default is already different than 1). If you'd like to run the application using Arduino, change the ports used by the servo motors in the "articulation_dict" dictionary inside the apply-model.py and the specify the arduino_mode 1, e.g.:
 ```bash
 $ python apply-model.py --arduino_mode 1
 ```
 
 5. When you're done, to quit the opened window just select it and press "Q".
 
 ## Project Structure
 <pre>
├── README.md
├── apply-model.py
├── collect-train-data.py
├── data
│   ├── gesture-label.csv
│   └── training-data.csv
├── helpers.py
├── model
│   └── clf.pkl
├── model-training.ipynb
├── reports
│   └── monography.pdf
└── requirements.txt
</pre>

#### apply-model.py
It's the main program of the application, which can be run using an Arduino or not. 

#### collect-train-data.py
It's the python script used to collect training data for the MLP model training. It is responsible to pre process the coordinates and save then into the **data/training-data.csv** file. 

#### model-training.ipynb
It's a notebook that trains a MLP model using the training-data.csv obtained by the collect-train-data.py. The output of this notebook is the **model/clf.pkl** file, which is imported by the apply-model to identify the hand gesture.  

#### gesture-label.csv
It's a csv file mapping the gestures that the model will identify to numbers, as the MLP model uses numbers as output. The labeling filled in this file will be shown in the image processing. 

#### helpers.py
It's a helper file containing functions used by both apply-model.py and collect-train-data.py.

 ## Model Training

Although the project already contains a trained model for the gestures listed in the project overview, it's possible to use the **collect-train-data.py**, **model-training.ipynb** and **gesture-label.csv** to collect data of other gestures and create and train a new model using it. 
 
### Data Collection 

#### 1. Strategy

As discussed in the archictecure topic the input of the gesture recognition model is the output of the MediaPipe Hands solution, which is the coordinates in width, height and depth of the hand's joints. However for the recognition of gestures the depth is irrelevant because a gesture will be the same regardless of the evaluated depth, so the input data of the model will be the x and y coordinates of each of the 21 joints, totaling 42 entries to result in the classification of one of the six possible gestures, as shown in the example below:

![processing](https://user-images.githubusercontent.com/21988243/221391221-c6e90848-781c-4d33-9f4f-a8926ca4af81.png)

It's not possible to use the raw coordinates gotten from the MediaPipe as input for the model because the coordinates are global based on the position of the hand in the camera view and this way a same gesture will have completely different coordinates if we move the hand position, as shown in the example below for a joint coordinate in a same gesture:

<div align="center"><img src="https://user-images.githubusercontent.com/21988243/221391648-a74d1d7a-8d9f-417d-badf-d1dbc76b0a12.png" width="40%" height="auto"></div>

The input desired to train the model are coordinates that varies only when a gesture change occurs and one way to achieve this is to use relative coordinates. The strategy used to get the relative coordinates is to subtract all coordinates by the coordinates of the wrist, making the wrist coordinate the origin (0,0), and all other coordinates relative to it, as shown in the example below:

<div align="center"><img src="https://user-images.githubusercontent.com/21988243/221392433-d55259f9-be30-4b76-a6c6-14cfa489dd80.png" width="80%" height="auto"></div>

After obtaining the relative coordinates, a good practice to improve the model's performance is to normalize the input data. The objective of normalization is to change the values of the numeric columns to a common scale without causing distortion in the value ranges. This is useful for the perceptron neural network model in question, as it uses linear combinations of inputs and associates weights to them. Since the coordinates can have several distinct value ranges, such as the Y-coordinate of the middle finger having values in the range of 0 to 100 and the thumb having values in the range of 300 to 400, the thumb coordinate can significantly influence the result due to its larger values, not necessarily because it is more important as a predictor. That being said, the method used to normalize the coordinates is the min-max normalization, in which all variables are placed in values between 0 and 1 using the following formula:

$$
coordinate_{scaled}  = \dfrac{coordinate - coordinate_{min}}{coordinate_{max} - coordinate_{min}}
$$

Below there is an example of applying this normalization to the relative coordinates of a hand gesture:

<div align="center"><img src="https://user-images.githubusercontent.com/21988243/221393405-5b3f995f-ee9f-4b2f-b6fc-5535aba0da7f.png" width="70%" height="auto"></div>

#### 2. How to Collect

As the model uses numeric values to be trained, the first step is to update the **gesture-label.csv** file with the mapping between the gesture and a number. For the model trained it was used this mapping:

<div align="center">
<table>
  <thead>
    <tr>
      <th style="text-align:center">Label</th>
      <th style="text-align:center">Gesture</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center">0</td>
      <td style="text-align:center">Wrist</td>
    </tr>
    <tr>
      <td style="text-align:center">1</td>
      <td style="text-align:center">Thumb</td>
    </tr>
    <tr>
      <td style="text-align:center">2</td>
      <td style="text-align:center">Index</td>
    </tr>
    <tr>
      <td style="text-align:center">3</td>
      <td style="text-align:center">Middle</td>
    </tr>
    <tr>
      <td style="text-align:center">4</td>
      <td style="text-align:center">Ring</td>
    </tr>
    <tr>
      <td style="text-align:center">5</td>
      <td style="text-align:center">Pinky</td>
    </tr>
  </tbody>
</table>
</div>

Then in the root folder of the project, run the following command:

```bash
$ python collect-train-data.py 
```

If it don't open a new window, maybe it's because your webcam is under other device number. Just as the main application, you can specify the device number, like for example ```python collect-train-data.py --device 1```.

To collect the data, you have the following options:

- Close the window: press the key **'Q'**
- Erase all training data: press the key **'E'**
- Save coordinate: press key from **0** to **9**

The gesture coordinates will be associated to the number used to save it and this number must match what was filled in the **gesture-label.csv**. Below is shown an example of a gesture made and the coordinates saved from that gesture:

<div align="center"><img src="https://user-images.githubusercontent.com/21988243/221394193-9fdab623-8152-440d-ad62-a09caea9fa6d.png" width="50%" height="auto"></div>

#### 3. Training

The training of the model is done using the [model-training.ipynb](https://github.com/Brunocds/computer-vision-robot-control/blob/main/model-training.ipynb) notebook. After collecting the data you can just run all the cells of the notebook to generate the new model pickle file. 

To train the model it was used the Python library scikit-learn, which has several tools for machine learning. Within it, the MLPClassifier model was used, which implements the multi-layer perceptron neural network model. When using this model, the following parameters were defined:

- **Hidden layer sizes:** 20, 15, 13, 10 and 8 
- **Activation function:** relu
- **Solver:** Stochastic Gradient Descent

<p style="margin-top: 100px; margin-bottom: 100px;">
<div align="center">
<img src="https://user-images.githubusercontent.com/21988243/221394795-5ad4fddb-528e-4d71-9fc3-8c902d11a4ab.png" width="100%" height="auto">
</div>
</p>

The training dataset consists of 651 coordinate records:

- **Wrist gesture:** 106 records
- **Thumb gesture:** 65 records 
- **Index gesture:** 96 records 
- **Middle gesture:** 109 records 
- **Ring gesture:** 155 records 
- **Pinky gesture:** 120 records

The data capture includes performing the gestures at different depths, inclinations, and small rotations, so that it is easier for the operator to have the desired gesture identified. For training the model, 80% of the data from the training dataset was used for training (521 records) and 20% for testing (130 records). After the training the model had the following confusion matrix, performing really well: 

<div align="center"><img src="https://user-images.githubusercontent.com/21988243/221394505-49cc5955-4361-45d2-91e1-dbc38920f4ce.png" width="40%" height="auto"></div>

## The Control of Joint Expansion and Contraction

The calculation of the distance between the thumb and the index finger is used to generate a percentage from 0% to 100% indicating how much the joint will expand or contract. The distance is calculated using the Pythagorean theorem, as shown in the image below:

<div align="center"><img src="https://user-images.githubusercontent.com/21988243/221395118-2c09ed3a-0d11-4406-acba-b9a7e1adf223.png" width="40%" height="auto"></div>

We have that this distance is given as a scalar number in pixels. To make it a percentage from 0% to 100%, with 0% when the fingers are touching and 100% something close to the maximum distance that the user can make between them, the distance between the wrist and the tip of the index finger was also calculated and then taken the quotient of both distances:

<div align="center"><img src="https://user-images.githubusercontent.com/21988243/221395302-fc19f24c-a59b-404d-bedc-c9ab0c9c2960.png" width="40%" height="auto"></div>

For cases where the distance between the tip of the thumb and index finger exceeds the distance between the wrist and the tip of the index finger, a value of 100% is forced as output. The idea of using this ratio as output is motivated by leaving a generic percentage for any hand that is used, since they are relative distances to the operator's hand, and also because there is no maximum distance between the tip of the thumb and index finger, varying from person to person how much they can be separated.
