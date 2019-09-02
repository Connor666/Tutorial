import os
from PIL import Image
import numpy as np
import tensorflow as tf
import face_recognition
import cv2

#文件路径设置（address）
data_dir_real='train/real/'
data_dir_fake='train/fake/'
test='video/test.mp4'
output_test='train/test_output/'
model_path='train/'
train= True # Ture: Training mode ，False: Validation mode


# 读取训练集（Load training set)
def read_data(data_dir_real,data_dir_fake):
    datas = []
    labels = []
    fpaths = []
    # Read data of real
    for fname in os.listdir(data_dir_real):
        fpath = os.path.join(data_dir_real, fname) # Record image name and address
        fpaths.append(fpath)
        image = Image.open(fpath) # Open image
        data = np.array(image) / 255.0 # 归一化(Normalization)
        datas.append(data)
        labels.append(0) # Take real as 0
    # Read data of fake
    for fname in os.listdir(data_dir_fake):
        fpath = os.path.join(data_dir_fake, fname)
        fpaths.append(fpath)
        image = Image.open(fpath)
        data = np.array(image) / 255.0 # 归一化(Normalization)
        datas.append(data)
        labels.append(1) # Take fake as 1

    datas = np.array(datas)
    labels = np.array(labels)

    return fpaths, datas,labels

fpaths, datas,labels = read_data(data_dir_real,data_dir_fake) # Load training set


# Placeholder
datas_placeholder = tf.placeholder(tf.float32,[None, 50,50, 3])
labels_placeholder = tf.placeholder(tf.int32, [None])
dropout_placeholdr = tf.placeholder(tf.float32)

# First layer and pooling, 20 kernel, size of kernel is 5
conv0 = tf.layers.conv2d(datas_placeholder, 20, 5, activation=tf.nn.relu)
# pooling filter is 2x2，stride is 2x2
pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])

# Second layer
conv1 = tf.layers.conv2d(pool0, 40, 4, activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])


# Fully Connected Layer
flatten = tf.layers.flatten(pool1)
fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)

# DropOut to prevent overfitting
dropout_fc = tf.layers.dropout(fc, dropout_placeholdr)
logits = tf.layers.dense(dropout_fc, 2,activation=None)
predicted_labels = tf.arg_max(logits, 1)

# Mean loss
losses = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(labels_placeholder, 2),
    logits=logits)
mean_loss = tf.reduce_mean(losses)

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(losses)

saver = tf.train.Saver()

with tf.Session() as sess:
    if train:
        print('Training')
        sess.run(tf.global_variables_initializer()) # initialization
        train_feed = {
            datas_placeholder: datas,
            labels_placeholder: labels,
            dropout_placeholdr: 0.5
        } # feed data
        for step in range(150): # iterate 150
            _, mean_loss_val = sess.run([optimizer, mean_loss],feed_dict=train_feed)
            if step % 10 == 0:
                print("step = {}\tmean loss = {}".format(step,mean_loss_val)) # print mean loss every 10 step
        saver.save(sess, model_path) # save model
        print('Training finished，save')

    else:
        print('Validation')
        vs = cv2.VideoCapture(test) # Load video
        saver.restore(sess, model_path) # load model
        read = 0
        while True:
            (grabbed, frame) = vs.read()
            if not grabbed:
                print('This frame is not grabbed')
                break
            read += 1

            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame) # Find faces

            for face_location in face_locations:
                datas = []

                top, right, bottom, left = face_location # Obtain face location
                cropped = frame[top:bottom, left:right]
                img_name = output_test + 'test_%d.png' % (read)
                cv2.imwrite(img_name, cropped, [int(cv2.IMWRITE_PNG_COMPRESSION), 9]) # Cut face images
                img = Image.open(img_name) # Open Img
                new_image = img.resize((50, 50), Image.BILINEAR) # Resize to 50*50
                new_image.save(os.path.join(output_test, os.path.basename(img_name))) # save resize img
                data = np.array(new_image) / 255.0
                datas.append(data)
                datas = np.array(datas)

                label_name_dict = {
                    0: "Real",
                    1: "Fake"
                } # Create dir
                test_feed_dict = {
                    datas_placeholder: datas,
                    dropout_placeholdr: 0
                }
                predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict) # Feed data

                predicted_label_name = label_name_dict[int(predicted_labels_val)] # Label transformation

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3) # Create rectangle to label face
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, predicted_label_name, (left + 10, bottom), font, 2, (255, 255, 255)) # put text of face result

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): # Type q to quit
                break
