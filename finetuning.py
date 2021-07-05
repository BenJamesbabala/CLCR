import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from FT_generator import DataGenerator
from FT_model import attention_unet, attention_unet_refined
from metrics import *
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, MeanIoU
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard

# tf.random.set_seed(100)
# np.random.seed(100)

## Path
model_path = 'path to saved finetuned weights' # Saving location
log_path = 'path to store training logs'

weights_paths = [] #Path to pretrained weights

train_path = "path to training set"
valid_path = "path to validation set"

## Parameters
image_size = (320, 256) # Original = (2448, 1920)
batch_size = 6
mode = 'seg'
target_classes = ["Good Crypts", "Good Villi", "Epithelium", "Brunner's Gland"] # Class names for Celiac Disease
filter_classes = ["Brunner's Gland"]    # oversampling images with classes contained in the list
lr = 1e-4
epochs = 500

encoder, att_unet, localizer, anti_celiac, masks_input_tensor = attention_unet_refined(input_shape=image_size, 
                                                                                        mask_channels=3,
                                                                                        out_channels=len(target_classes), 
                                                                                        multiplier=10, 
                                                                                        freeze_encoder=False,
                                                                                        freeze_decoder=False, 
                                                                                        use_constraints = False,
                                                                                        dropout_rate=0.0)

if mode == 'seg':
    model = att_unet
    #model = UEfficientNet(input_shape=(320,256,3),dropout_rate=0.20, output_classes=len(target_classes))
    metrics = [uniclass_dice_coeff_0, uniclass_dice_coeff_1, uniclass_dice_coeff_2, uniclass_dice_coeff_3, multiclass_dice_coeff]
    # losses = multiclass_dice_loss(loss_scales=[1., 1., 1., 1.])
    losses = focal_tversky_loss
elif mode == 'loc':
    model = localizer
    metrics = [focal_loss, bifurcated_mse]
    # losses = constraint_focal_mse_loss(image_size, 16, batch_size, factor=20.0) 
    losses = focal_mse_loss
elif mode == 'full':
    model = anti_celiac
    metrics = {'out': [multiclass_dice_coeff, 'acc'], 'concat0': [focal_loss, bifurcated_mse]}
    losses = {'out': multiclass_dice_loss(loss_scales=[2, 1, 3]), 'concat0': focal_mse_loss}

optimizer = Adam(learning_rate=lr)
model.compile(loss=losses, optimizer=optimizer, metrics=metrics)
model.summary()

# Resume from checkpoint
if weights_paths != []:
    for wp in weights_paths:
        model.load_weights(wp, by_name=True, skip_mismatch=True)
        print("loaded weights")

if encoder_weights is not None:
    temp_model = tf.keras.models.clone_model(model)
    temp_model.set_weights(model.get_weights())
    encoder.load_weights(encoder_weights, by_name=True)

# Data generators
train_generator = DataGenerator(train_path, image_size, batch_size, mode, target_classes, filter_classes=filter_classes, augment=True)
val_generator = DataGenerator(valid_path, image_size, batch_size, mode, target_classes, augment=False)


callbacks = [ModelCheckpoint(model_path, save_weights_only=True),
            ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, min_lr=1e-6),
            EarlyStopping(monitor='loss', patience=25, restore_best_weights=False), 
            TensorBoard(log_dir=log_path, update_freq='epoch', write_graph=False, profile_batch=0),
]

model.fit(x = train_generator
        , steps_per_epoch = train_generator.__len__()
        , epochs = epochs
        , verbose = 1
        , callbacks = callbacks
        , validation_data = val_generator
        , validation_steps = val_generator.__len__()
        , workers = 4
        , validation_freq = 5
        , max_queue_size = 20
        , initial_epoch = 0)
