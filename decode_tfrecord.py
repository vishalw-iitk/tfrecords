import tensorflow as tf
import os
import shutil
import matplotlib.image as mpimg
import numpy as np
import cv2


sess = tf.InteractiveSession()

reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(['/home/vishal/Desktop/sentence_cmle_pipeline-00000-of-00001.tfrecord'])

_, serialized_example = reader.read(filename_queue)

context_features = {
	'decoder_input_data': tf.FixedLenFeature([], dtype=tf.string),
    'len': tf.FixedLenFeature([], dtype=tf.int64),
    'height': tf.FixedLenFeature([], dtype=tf.int64),
    'width': tf.FixedLenFeature([], dtype=tf.int64),
    'depth': tf.FixedLenFeature([], dtype=tf.int64)
}
sequence_features = {
	'label': tf.FixedLenSequenceFeature([], dtype=tf.string),
    'frames': tf.FixedLenSequenceFeature([],dtype=tf.string),
}

context_data, sequence_data = tf.parse_single_sequence_example(
    serialized=serialized_example,
    context_features=context_features,
    sequence_features=sequence_features)

#cont = tf.decode_raw(context_data['label'], tf.float32)

seqd = tf.decode_raw(sequence_data['label'], tf.float32)
#seqd = tf.reshape(seqd, [context_data['len'], 60,60, 4])


tf.train.start_queue_runners(sess)

print(seqd.eval().shape)
# cv2.imwrite('image0.jpg',seqd.eval()[0][:,:,0]*255)
# cv2.imwrite('image1.jpg',seqd.eval()[1][:,:,1]*255)
# cv2.imwrite('image2.jpg',seqd.eval()[2][:,:,2]*255)
# cv2.imwrite('image3.jpg',seqd.eval()[3][:,:,3]*255)