import argparse
import cv2
import numpy as np
import sys

import socket
import json
import paho.mqtt.client as mqtt
from random import randint

# from handle_models import handle_output, preprocessing
from inference_server_comm_attention_tracking import Network

INPUT_STREAM = "head-pose-face-detection-female-and-male.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
ADAS_MODEL = "/home/workspace/models/facial-landmarks-35-adas-0002.xml"

CAR_COLORS = ["white", "gray", "yellow", "red", "green", "blue", "black"]
CAR_TYPES = ["car", "bus", "truck", "van"]

HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

def get_args():
    '''
    Gets the arguments from the command line.
    '''

    parser = argparse.ArgumentParser("Basic Edge App with Inference Engine")
    # -- Create the descriptions for the commands

    c_desc = "CPU extension file location, if applicable"
    d_desc = "Device, if not CPU (GPU, FPGA, MYRIAD)"
    i_desc = "The location of the input image"
    m_desc = "The location of the model XML file"
    t_desc = "The type of model: POSE, TEXT or CAR_META"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    required.add_argument("-m", help=m_desc, default=None)
    required.add_argument("-t", help=t_desc, default=None)
    optional.add_argument("-c", help=c_desc, default=None)
    optional.add_argument("-d", help=d_desc, default="CPU")
    args = parser.parse_args()

    return args


def get_mask(processed_output):
    '''
    Given an input image size and processed output for a semantic mask,
    returns a masks able to be combined with the original image.
    '''
    # Create an empty array for other color channels of mask
    empty = np.zeros(processed_output.shape)
    # Stack to make a Green mask where text detected
    mask = np.dstack((empty, processed_output, empty))

    return mask

def draw_masks(result, width, height):
    '''
    Draw semantic mask classes onto the frame.
    '''
    # Create a mask with color by class
    classes = cv2.resize(result[0].transpose((1,2,0)), (width,height), 
        interpolation=cv2.INTER_NEAREST)
    unique_classes = np.unique(classes)
    out_mask = classes * (255/20)
    
    # Stack the mask so FFmpeg understands it
    out_mask = np.dstack((out_mask, out_mask, out_mask))
    out_mask = np.uint8(out_mask)

    return out_mask, unique_classes


def get_class_names(class_nums):
    class_names= []
    for i in class_nums:
        class_names.append(CLASSES[int(i)])
    return class_names



def create_output_image(model_type, image, output):
    '''
    Using the model type, input image, and processed output,
    creates an output image showing the result of inference.
    '''
    if model_type == "POSE":
        # Remove final part of output not used for heatmaps
        output = output[:-1]
        # Get only pose detections above 0.5 confidence, set to 255
        for c in range(len(output)):
            output[c] = np.where(output[c]>0.5, 255, 0)
        # Sum along the "class" axis
        output = np.sum(output, axis=0)
        # Get semantic mask
        pose_mask = get_mask(output)
        # Combine with original image
        image = image + pose_mask
        return image
    elif model_type == "TEXT":
        # Get only text detections above 0.5 confidence, set to 255
        output = np.where(output[1]>0.5, 255, 0)
        # Get semantic mask
        text_mask = get_mask(output)
        # Add the mask to the image
        image = image + text_mask
        return image
    elif model_type == "CAR_META":
        # Get the color and car type from their lists
        color = CAR_COLORS[output[0]]
        car_type = CAR_TYPES[output[1]]
        # Scale the output text by the image shape
        scaler = max(int(image.shape[0] / 1000), 1)
        # Write the text of color and type onto the image
        image = cv2.putText(image, 
            "Color: {}, Type: {}".format(color, car_type), 
            (50 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 
            2 * scaler, (255, 255, 255), 3 * scaler)
        return image
    else:
        print("Unknown model type, unable to create output image.")
        return image

def perform_inference(args, model):
    ### TODO: Connect to the MQTT server
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    '''
    Performs inference on an input image, given a model.
    '''
    # Create a Network for using the Inference Engine
    inference_network = Network()

    # self added 1 line
    face_cascade = cv2.CascadeClassifier("/opt/intel/openvino/opencv/etc/haarcascades/haarcascade_frontalface_default.xml")

    # Load the model in the network, and obtain its input shape
    n, c, h, w = inference_network.load_model(model, args.d, CPU_EXTENSION)

    # self added 5 lines
    net_input_shape = inference_network.get_input_shape()
    # 0 following line is for webcam 
    # cap = cv2.VideoCapture(0) 
    cap = cv2.VideoCapture(args.i)
    cc= cap.open(args.i)

    counter = 0
    #incident_flag = False

    # print(net_input_shape)
    model_width = int(net_input_shape[2])
    model_height = int(net_input_shape[3])
    
    # Read the input image
    
    # out = cv2.VideoWriter('out_face_detection_video_aync_inference.mp4', 0x00000021, 10, (100, 100))

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))
    # print(width, height)
    out = cv2.VideoWriter('out_face_detection_video_aync_inference.mp4', 0x00000021, 10, (width, height))
   
    speed = 0
    while cap.isOpened():
        # Read the next frame
        flag, image = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        counter += 1

        counter_each = []

        ### TODO: Preprocess the input image
        # preprocessed_image = preprocessing(image, h, w)
    
        # self added 
        faces = face_cascade.detectMultiScale(image, 1.2, 2) 
        for i in range(len(faces)): 
            counter_each.append(0)
        #if faces is not None:
        if len(faces) != 0:
            # print(faces)
            '''
            if not incident_flag:
                timestamp = counter / 30
                print("Log: paying attention at {:.2f} seconds.".format(timestamp))
            '''
            # original  / 1000 
            scaler = max(int(image.shape[0] / 2000), 1) 
            color = (0, 0, 255)
            # Write the text of color and type onto the image
            ind = 0 
            incident_flag = [] 
            for i in range(len(faces)):           
               incident_flag.append(False)
               counter_each[i] = counter_each[i] + 1

            for (ix, iy, w, h) in faces:  
               if not incident_flag[ind]:
                   timestamp = counter / 30
                   print("Log: person {} paying attention at {:.2f} seconds.".format(ind+1, timestamp)) 
               incident_flag[ind] = True  
               ind = ind + 1          
               # image = cv2.putText(image, "Good!", ((ix) * scaler, (iy-50) * scaler), cv2.FONT_HERSHEY_SIMPLEX, 2 * scaler, (255, 255, 255), 3 * scaler)
               image = cv2.putText(image, "Good!", ((ix) * scaler, (iy-30) * scaler), cv2.FONT_HERSHEY_SIMPLEX, scaler, (0, 255, 255), scaler)
               #Crop face detected
               face_image = image[iy:iy+h, ix:ix+w]

               # Draw rectangle around face
               thickness = 2
               color = (255, 0, 0)
               image = cv2.rectangle(image, (ix, iy), (ix+w, iy+h), color, thickness)

               face_width = face_image.shape[1]
               face_height = face_image.shape[0]

               #Resize cropped face to match IR input size
               face_image = cv2.resize(face_image, (net_input_shape[3], net_input_shape[2]))
               face_image = face_image.transpose((2, 0, 1))
               face_image = face_image.reshape(1, *face_image.shape)

               # Perform synchronous inference on the image
               # inference_network.sync_inference(face_image)
               # asynchronous inference
               inference_network.async_inference(face_image)
               if inference_network.wait() == 0:              
                   result = inference_network.extract_output()
                   for i in range(0, result.shape[1], 2):
                       x, y = int(ix+result[0][i]*face_width), iy+int(result[0][i+1]*face_height)
                       # Draw Facial key points
                       cv2.circle(image, (x, y), 2, color, thickness)

        # class_names = get_class_names(classes)
        class_names = [] 
        index = 0   
        #speed = 0
        for f in faces:
            class_names.append("person"+str(index+1))
            speed = speed + counter_each[index]
            index = index + 1
        #speed = randint(50,70)
       
         
        ### TODO: Send the class names and speed to the MQTT server
        ### Hint: The UI web server will check for a "class" and
        ### "speedometer" topic. Additionally, it expects "class_names"
        ### and "speed" as the json keys of the data, respectively.
        client.publish("class", json.dumps({"class_names": class_names}))
        client.publish("speedometer", json.dumps({"speed": speed}))

        #if len(faces) == 0:
            #incident_flag = False
            #print("Not paying attention!")
        
        out.write(image)
        ### TODO: Send frame to the ffmpeg server
        sys.stdout.buffer.write(image)
        # sys.stdout.flush()

        # Break if escape key pressed
        if key_pressed == 27:
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    ### TODO: Disconnect from MQTT
    client.disconnect()


    #output_func = handle_output(args.t)
    #processed_output = output_func(output, image.shape)

    # Create an output image based on network
    #output_image = create_output_image(args.t, image, processed_output)

    # Save down the resulting image
    # cv2.imwrite("outputs/{}-output-attention-tracking.png".format(args.t), image)


def main():
    args = get_args()
    model = ADAS_MODEL
    perform_inference(args, model)


if __name__ == "__main__":
    main()
