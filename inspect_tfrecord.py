import tensorflow as tf

# for example in tf.python_io.tf_record_iterator("/home/vishal/Desktop/tf_data_path/Apache_beam_records_-00000-of-00020.tfrecord"):
#     print(tf.train.Example.FromString(example), file=open("new_tftext.txt", "a"))

for example in tf.python_io.tf_record_iterator("/home/vishal/Desktop/Apache_beam_records_Test_records_sorted_2-00000-of-00001"):
    print(tf.train.SequenceExample.FromString(example), file=open("csv_wise-1.txt", "a"))
