#!/usr/bin/env python
#
# Copyright (c) 2011, Willow Garage, Inc.
# All rights reserved.
#
# Software License Agreement (BSD License 2.0)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of {copyright_holder} nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Darby Lim

import os
import select
import sys
import rclpy


from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile

import mediapipe as mp 
import cv2 
import numpy as np 
import pandas as pd
import joblib
from helpers import get_handedness, pre_process_hand_landmarks, get_args

import subprocess


################CV#########################
args = get_args()

# Object that let us draw landmarks in our image 
mp_drawing = mp.solutions.drawing_utils

# Object with all hand tracking methods of mediapipe
mp_hands = mp.solutions.hands

# Loading gesture recognition model 
gesture_classifier = joblib.load('model/clf.pkl')

# The cv2.VideoCapture needs an number representing which device will be used 
# if the program does not work for you, try specifying a device other than 0 
device = args.device
# Object that reads from the webcam
cap = cv2.VideoCapture(device)

# Get the width and height so we can draw on the image using opencv
video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Hand gesture label map 
gesture_label = pd.read_csv('data/gesture-label.csv', encoding = 'latin1')
################CV#########################


if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty

BURGER_MAX_LIN_VEL = 0.22
BURGER_MAX_ANG_VEL = 2.84

WAFFLE_MAX_LIN_VEL = 0.26
WAFFLE_MAX_ANG_VEL = 1.82

LIN_VEL_STEP_SIZE = 0.01
ANG_VEL_STEP_SIZE = 0.1

TURTLEBOT3_MODEL = os.environ['TURTLEBOT3_MODEL']

msg = """
Control Your TurtleBot3!
---------------------------
Moving around:
        w
   a    s    d
        x

w/x : increase/decrease linear velocity (Burger : ~ 0.22, Waffle and Waffle Pi : ~ 0.26)
a/d : increase/decrease angular velocity (Burger : ~ 2.84, Waffle and Waffle Pi : ~ 1.82)

space key, s : force stop

CTRL-C to quit
"""

e = """
Communications Failed
"""

# Reads single character from console. 
# CHANGE THIS TO GESTURE INPUTS.
def get_key(settings):
    if os.name == 'nt':
        return msvcrt.getch().decode('utf-8')
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

# NO NEED TO CHANGE.
# Prints current linear and angular velocities. 
def print_vels(target_linear_velocity, target_angular_velocity):
    print('currently:\tlinear velocity {0}\t angular velocity {1} '.format(
        target_linear_velocity,
        target_angular_velocity))

# NO NEED TO CHANGE.
# implements a simple proportional control mechanism. 
# Adjusts the output value based on the difference between the current input and the desired output, with a given "slop" or tolerance.
def make_simple_profile(output, input, slop):
    if input > output:
        output = min(input, output + slop)
    elif input < output:
        output = max(input, output - slop)
    else:
        output = input

    return output

# NO NEED TO CHANGE.
# Restricts input variable to a specified range
def constrain(input_vel, low_bound, high_bound):
    if input_vel < low_bound:
        input_vel = low_bound
    elif input_vel > high_bound:
        input_vel = high_bound
    else:
        input_vel = input_vel

    return input_vel

# NO NEED TO CHANGE.
# Ensures that linear velocities are within specific bounds
def check_linear_limit_velocity(velocity):
    if TURTLEBOT3_MODEL == 'burger':
        return constrain(velocity, -BURGER_MAX_LIN_VEL, BURGER_MAX_LIN_VEL)
    else:
        return constrain(velocity, -WAFFLE_MAX_LIN_VEL, WAFFLE_MAX_LIN_VEL)

# NO NEED TO CHANGE.
# Ensures that angular velocities are within specific bounds
def check_angular_limit_velocity(velocity):
    if TURTLEBOT3_MODEL == 'burger':
        return constrain(velocity, -BURGER_MAX_ANG_VEL, BURGER_MAX_ANG_VEL)
    else:
        return constrain(velocity, -WAFFLE_MAX_ANG_VEL, WAFFLE_MAX_ANG_VEL)
    
def play_mp3(file_paths):
    try:
        subprocess.run(["mpg123", file_paths])
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    settings = None
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

    rclpy.init()

    qos = QoSProfile(depth=10)
    # Creates a ROS2 node
    node = rclpy.create_node('teleop_keyboard')
    # Creates a publisher for the /cmd_vel topic with the Twist message type.
    pub = node.create_publisher(Twist, 'cmd_vel', qos)

    # Initialization to start variables from 0.
    status = 0
    target_linear_velocity = 0.0
    target_angular_velocity = 0.0
    control_linear_velocity = 0.0
    control_angular_velocity = 0.0

# LINK GESTURE INPUTS FROM LINE 83, MODIFY ACCORDINGLY.
    try:
        print(msg)
        text_label_predicted = ""
        relative_distance_thumb_index = 0.0
        # angle = 0.0
        state_machine = 0
        audio = ""
        with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.5
        ) as hands:
            while cap.isOpened():
                # cap.read() return two variables, the 'results' which is a boolean
                # identifying if the image was read and the frame that is a cv2 image
                # object of the frame captured
                ret, frame = cap.read()

                # Before processing our image with mediapipe is necessary to convert it 
                # from BGR to RGB, because mediapipe works with RGB and opencv with BGR
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Flip on horizontal so the lib detects correct handness
                image = cv2.flip(image, 1)

                # Setting the writable flag to false before process with mediapipe leads 
                # to improvement in the performance 
                image.flags.writeable = False

                # Do the actual processing with the mediapipe lib 
                results = hands.process(image)

                # Setting back the flag of writable so we can draw in the image
                image.flags.writeable = True

                handedness_detected = [] 
                
                # Drawing landmarks to the image
                ## If any hand was detected
                if results.multi_hand_landmarks:           
                    # The enumerate is used to multi hand detection, the hand variable
                    # is basically all the landmarks from one hand
                    for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                                        
                        # Get handedness label
                        handedness_label = get_handedness(
                            hand_index = hand_index,
                            results = results
                        )
                        
                        handedness_detected.append(handedness_label)
                        
                        # Draw bounding box only to the left hand
                        if handedness_label=='Left': 
                            x_max = 0
                            y_max = 0
                            x_min = video_width
                            y_min = video_height
                            for coordinates in hand_landmarks.landmark:
                                x, y = int(coordinates.x * video_width), int(coordinates.y * video_height)
                                if x > x_max:
                                    x_max = x
                                if x < x_min:
                                    x_min = x
                                if y > y_max:
                                    y_max = y
                                if y < y_min:
                                    y_min = y
                            cv2.rectangle(image, (x_min-10, y_min-10), (x_max+10, y_max+10), (50, 50, 50), 2)
                            cv2.rectangle(image, (x_min-10, y_min-10), (x_max+10, y_min-50), (50, 50, 50), -1)
                                
                            # Pre process the hand landmarks coordinates 
                            processed_hand_landmarks = pre_process_hand_landmarks(hand_index, results, video_width, video_height)

                            # Predicting the gesture label
                            label_predicted = int(gesture_classifier.predict(processed_hand_landmarks.reshape(1,-1))[0])
                            text_label_predicted = gesture_label[gesture_label.Label==label_predicted].Gesture.to_list()[0]
                            
                            # Updating the gesture with the text_label_predicted
                            gesture = text_label_predicted
                            # print(text_label_predicted)
                            ###############################################################################output hand####################
                            # Writing the predicted label to the frame 
                            image = cv2.putText(
                                img = image, 
                                text = text_label_predicted,
                                org = (x_min+5,y_min-20), #coordinates
                                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale = 0.7, 
                                color = (255, 255, 255), #RGB
                                thickness = 2, 
                                lineType = cv2.LINE_AA
                            )

                            # Utility used to draw the image based on the landmark values
                            mp_drawing.draw_landmarks(
                                image = image,
                                landmark_list = hand_landmarks,
                                connections = mp_hands.HAND_CONNECTIONS,
                                landmark_drawing_spec = mp_drawing.DrawingSpec(
                                                            color = (1, 190, 255),
                                                            thickness = 2,
                                                            circle_radius = 4
                                                        ),
                                connection_drawing_spec = mp_drawing.DrawingSpec(
                                                            color = (86, 213, 0),
                                                            thickness = 2,
                                                            circle_radius = 2
                                                        )
                            )
                        elif handedness_label=='Right':                
                            # Drawing a circle in the joints of the THUMB_TIP and INDEX_FINGER_TIP and
                            # storing its coordinates 
                            joints = ['THUMB_TIP', 'INDEX_FINGER_TIP']
                            joints_coordinates = {}
                            for joint in joints:
                                coordinates = {}
                                coordinates['x'] = int(results.multi_hand_landmarks[hand_index].landmark[mp_hands.HandLandmark[joint]].x*video_width)
                                coordinates['y'] = int(results.multi_hand_landmarks[hand_index].landmark[mp_hands.HandLandmark[joint]].y*video_height)
                                joints_coordinates[joint] = coordinates
                                # Drawing circle
                                image = cv2.circle(
                                    img = image, 
                                    center = (coordinates['x'],coordinates['y']), 
                                    radius = 4, 
                                    color = (148, 0, 211), 
                                    thickness = 2
                                )
                                # Drawing white border for the circle
                                image = cv2.circle(
                                    img = image, 
                                    center = (coordinates['x'],coordinates['y']), 
                                    radius = 6, 
                                    color = (255, 255, 255), 
                                    thickness = 1
                                )
                            
                            # Draw line between the thumb_tip and index_finger_tip
                            image = cv2.line(
                                img = image, 
                                pt1 = (joints_coordinates['THUMB_TIP']['x'], joints_coordinates['THUMB_TIP']['y']), 
                                pt2 = (joints_coordinates['INDEX_FINGER_TIP']['x'], joints_coordinates['INDEX_FINGER_TIP']['y']), 
                                color = (255,4,163), 
                                thickness = 2
                            )
                            
                            # Drawing circle in the center of the line
                            min_x = min(joints_coordinates['THUMB_TIP']['x'],joints_coordinates['INDEX_FINGER_TIP']['x'])
                            min_y = min(joints_coordinates['THUMB_TIP']['y'],joints_coordinates['INDEX_FINGER_TIP']['y'])
                            diff_x = int(abs(joints_coordinates['THUMB_TIP']['x'] - joints_coordinates['INDEX_FINGER_TIP']['x'])/2)
                            diff_y = int(abs(joints_coordinates['THUMB_TIP']['y'] - joints_coordinates['INDEX_FINGER_TIP']['y'])/2)
                            image = cv2.circle(
                                img = image, 
                                center = (min_x + diff_x, min_y + diff_y), 
                                radius = 6, 
                                color = (148, 0, 211), 
                                thickness = 3
                            ) 
                            
                            # Drawing white border for the circle in the center of the line
                            image = cv2.circle(
                                    img = image, 
                                    center = (min_x + diff_x, min_y + diff_y), 
                                    radius = 8, 
                                    color = (255, 255, 255), 
                                    thickness = 1
                                )
                            
                            # Draw the rectangle border that will contain the relative distance of the line 
                            # between the thumb_tip and index_finger_tip 
                            
                            beginning_x_coordinate_rect = 85
                            end_x_coordinate_rect = 485
                            
                            image = cv2.rectangle(
                                img = image,
                                pt1 = (beginning_x_coordinate_rect, 30),
                                pt2 = (end_x_coordinate_rect, 70),
                                color = (255,4,163),
                                thickness = 2
                            )
                            
                            # The relative distance will be the line between the thumb tip and index_finger_tip
                            # divided by the line between the wrist and index_finger_dip
                            wrist_coordinates = (results.multi_hand_landmarks[hand_index].landmark[mp_hands.HandLandmark['WRIST']].x * video_width, results.multi_hand_landmarks[hand_index].landmark[mp_hands.HandLandmark['WRIST']].y * video_height)
                            distance_wrist_index = np.sqrt(((wrist_coordinates[0]-joints_coordinates['INDEX_FINGER_TIP']['x'])**2) + ((wrist_coordinates[1]-joints_coordinates['INDEX_FINGER_TIP']['y'])**2))
                            distance_thumb_index = np.sqrt(((joints_coordinates['THUMB_TIP']['x']-joints_coordinates['INDEX_FINGER_TIP']['x'])**2) + ((joints_coordinates['THUMB_TIP']['y']-joints_coordinates['INDEX_FINGER_TIP']['y'])**2))
                            relative_distance_thumb_index = distance_thumb_index/distance_wrist_index
                            #Offset
                            relative_distance_thumb_index = relative_distance_thumb_index - 0.08
                            
                            if relative_distance_thumb_index > 1:
                                relative_distance_thumb_index = 1
                            elif relative_distance_thumb_index < 0:
                                relative_distance_thumb_index = 0
                            
                            linear_interpolation = int(beginning_x_coordinate_rect + ((end_x_coordinate_rect-beginning_x_coordinate_rect)*((relative_distance_thumb_index))))
                            
                            # Draw the rectangle that will fill the base rectangle according to the distance
                            # between the thumb_tip index_finger_tip
                            image = cv2.rectangle(
                                img = image,
                                pt1 = (beginning_x_coordinate_rect, 30),
                                pt2 = (linear_interpolation, 70),
                                color = (255,4,163),
                                thickness = -1
                            )

                            # Writing relative distance
                            image = cv2.putText(
                                img = image, 
                                text = str(int(relative_distance_thumb_index*100))+' %',
                                org = (end_x_coordinate_rect+15,55), #coordinates
                                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale = 0.8, 
                                color = (255,4,163), #RGB
                                thickness = 2, 
                                lineType = cv2.LINE_AA
                            )
                            
                          
                        key = get_key(settings)
                        if text_label_predicted == 'Forward':
                            target_linear_velocity =\
                                check_linear_limit_velocity(BURGER_MAX_LIN_VEL * relative_distance_thumb_index)
                            target_angular_velocity =\
                                check_angular_limit_velocity(float(0))
                            status = status + 1
                            print_vels(target_linear_velocity, target_angular_velocity)
                        elif text_label_predicted== 'Backward':
                            target_linear_velocity =\
                                check_linear_limit_velocity(-(BURGER_MAX_LIN_VEL * relative_distance_thumb_index))
                            target_angular_velocity =\
                                check_angular_limit_velocity(float(0))
                            status = status + 1
                            print_vels(target_linear_velocity, target_angular_velocity)
                        elif text_label_predicted== 'Left':
                            target_angular_velocity =\
                                check_angular_limit_velocity(BURGER_MAX_ANG_VEL * relative_distance_thumb_index)
                            target_linear_velocity =\
                                check_linear_limit_velocity(float(0))
                            status = status + 1
                            print_vels(target_linear_velocity, target_angular_velocity)
                        elif text_label_predicted == 'Right':
                            target_angular_velocity =\
                                check_angular_limit_velocity(-(BURGER_MAX_ANG_VEL * relative_distance_thumb_index))
                            target_linear_velocity =\
                                check_linear_limit_velocity(float(0))
                            status = status + 1
                            print_vels(target_linear_velocity, target_angular_velocity)
                        elif text_label_predicted == 'Stop' or key == 's':                                                  #E-STOP FEATURE
                            target_linear_velocity = 0.0
                            control_linear_velocity = 0.0
                            target_angular_velocity = 0.0
                            control_angular_velocity = 0.0
                            print_vels(target_linear_velocity, target_angular_velocity)
                        else:
                            if (key == '\x03'):
                                break
                        
                        if state_machine == 0:
                            if text_label_predicted == "Left":
                                print("im at left")
                                play_mp3("/home/arms/computer-vision-robot-control/Audio_Files/turning_left.mp3")
                            if text_label_predicted == "Right":
                                play_mp3("/home/arms/computer-vision-robot-control/Audio_Files/turning_right.mp3")
                            if text_label_predicted == "Backward":
                                play_mp3("/home/arms/computer-vision-robot-control/Audio_Files/reversing.mp3")
                            

                        elif state_machine == 30:
                            state_machine =-1
                        
                        state_machine +=1

                        if status == 20:
                            print(msg)
                            status = 0

                        twist = Twist()

                        control_linear_velocity = make_simple_profile(
                            control_linear_velocity,
                            target_linear_velocity,
                            (LIN_VEL_STEP_SIZE / 2.0))

                        twist.linear.x = control_linear_velocity
                        twist.linear.y = 0.0
                        twist.linear.z = 0.0

                        control_angular_velocity = make_simple_profile(
                            control_angular_velocity,
                            target_angular_velocity,
                            (ANG_VEL_STEP_SIZE / 2.0))

                        twist.angular.x = 0.0
                        twist.angular.y = 0.0
                        twist.angular.z = control_angular_velocity

                        pub.publish(twist)

                # Converting it back to BGR so we can display using opencv
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Display the output frame of the webcam
                cv2.imshow('Hand Tracking', image)                        
                
                # - If 'q' is pressed the window is closed;
                key = cv2.waitKey(10)
                if key==ord('q'):
                    break

    except Exception as e:
        print(e)

    finally:
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0

        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0

        pub.publish(twist)

        if os.name != 'nt':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
                            
        

        


if __name__ == '__main__':
    main()
