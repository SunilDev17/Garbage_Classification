# Garbage_Classification using Deep Learning
This project classifies garbage into different categories (e.g., plastic,cardboard,glass,trash,paper,etc)
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns 
import tensorflow as tf  
from tensorflow import keras


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import os

dataset=r"C:\Current\Desktop\Internship\TrashType_Image_Dataset"
image_size = (124, 124)
batch_size = 32
seed=42

train= tf.keras.utils.image_dataset_from_directory(
    dataset,
    validation_split=0.2,
    subset="training",
    seed=seed,
    shuffle = True,
    image_size=image_size,
    batch_size=batch_size
)

validate = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    shuffle = True,
    image_size=image_size,
    batch_size=batch_size
)
val_class= validate.class_names

val_batches = tf.data.experimental.cardinality(val_ds)  

test_dataset = val_ds.take(val_batches // 2)  

val_dataset = val_ds.skip(val_batches // 2)  

test_ds_eval = test_ds.cache().prefetch(tf.data.AUTOTUNE)  

print(train_ds.class_names)
print(val_class)
print(len(train_ds.class_names))




print(train.class_names)
print(val_class)
print(len(train_ds.class_names))
