import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Paths & config
# ----------------------------
# CNNs
train_dir = r"D:\DoAn\orange_egg_banana_train"   # <- change to your folders
val_dir   = r"D:\DoAn\orange_egg_banana_val"
IMG_SIZE  = (240, 240)
BATCH     = 32
NUM_CLASSES = 16   # or 5/6 as needed

# ----------------------------
# INPUT DATA (grayscale loaders)
# ----------------------------
train_datagen = ImageDataGenerator(
    # rescale=1./255,
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2)    # helpful for lighting variation
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    color_mode="grayscale",         # <--- GRAYSCALE INPUT
    batch_size=BATCH,
    class_mode="categorical",
    shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    color_mode="grayscale",         # <--- GRAYSCALE INPUT
    batch_size=BATCH,
    class_mode="categorical",
    shuffle=False
)

print("Class mapping:", train_generator.class_indices)

steps_per_epoch   = train_generator.samples // train_generator.batch_size
validation_steps  = val_generator.samples // val_generator.batch_size

# ----------------------------
# MODEL (pretrained backbone; convert 1ch -> 3ch inside)
# ----------------------------
def build_grayscale_transfer_model(num_classes=NUM_CLASSES):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), weights="imagenet"
    )
    base.trainable = False  # Stage 1: freeze backbone

    inp = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1), name='input_gray')           # grayscale input
    x = layers.Concatenate(axis=-1, name="gray_to_rgb")([inp, inp, inp])
    # x = layers.Rescaling(scale=1./127.5, offset=-1.0, name="to_minus1_1")(x)
    # x = layers.Lambda(lambda t: tf.image.grayscale_to_rgb(t))(inp) 

    # mild, safe augmentations
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.10)(x)
    x = layers.RandomTranslation(0.10, 0.10)(x)

    # x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inp, out)
    return model, base

model, base = build_grayscale_transfer_model()

# ----------------------------
# Compile + callbacks
# ----------------------------
model.compile(
    optimizer=AdamW(learning_rate=3e-4, weight_decay=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05), #0.05-0.02
    metrics=["accuracy"]
)

cbs_stage1 = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ModelCheckpoint("best_stage1.keras", monitor="val_loss", save_best_only=True),
]

# ----------------------------
# TRAIN — Stage 1 (frozen backbone)
# ----------------------------
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=cbs_stage1
)

# ----------------------------
# Fine-tune — Stage 2 (partial unfreeze + tiny LR)
# ----------------------------
base.trainable = True
for layer in base.layers[:-150]:   # unfreeze last ~60 layers; adjust if needed
    layer.trainable = False

model.compile(
    optimizer=AdamW(learning_rate=3e-5, weight_decay=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
    metrics=["accuracy"]
)

cbs_stage2 = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ModelCheckpoint("best_stage2.keras", monitor="val_loss", save_best_only=True),
]

history_ft = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=cbs_stage2
)

model.save(r"D:\DoAn\trash_16_cls_grayscale_finetuned_noLambda.keras")
print("Saved to D:\DoAn\trash_16_cls_grayscale_finetuned_noLambda.keras")

def plot_combined_history(history_stage1, history_stage2):
    """
    Plots the combined training and validation loss/accuracy from two stages.

    Args:
        history_stage1 (tf.keras.callbacks.History): History object from Stage 1.
        history_stage2 (tf.keras.callbacks.History): History object from Stage 2 (fine-tuning).
    """

    # Concatenate the metric lists from both history objects
    loss = history_stage1.history['loss'] + history_stage2.history['loss']
    val_loss = history_stage1.history['val_loss'] + history_stage2.history['val_loss']
    accuracy = history_stage1.history['accuracy'] + history_stage2.history['accuracy']
    val_accuracy = history_stage1.history['val_accuracy'] + history_stage2.history['val_accuracy']

    total_epochs = len(loss)
    epochs = range(1, total_epochs + 1)
    stage1_epochs = len(history_stage1.history['loss'])

    plt.figure(figsize=(14, 6))

    # --- Loss Plot ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    
    # Mark the transition between stages
    plt.axvline(x=stage1_epochs, color='k', linestyle='--', label='Start Fine-tuning')

    plt.title('Combined Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # Set x-ticks for better readability
    plt.xticks(np.arange(1, total_epochs + 1, max(1, total_epochs // 10))) 

    # --- Accuracy Plot ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')

    # Mark the transition between stages
    plt.axvline(x=stage1_epochs, color='k', linestyle='--', label='Start Fine-tuning')

    plt.title('Combined Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    # Set x-ticks for better readability
    plt.xticks(np.arange(1, total_epochs + 1, max(1, total_epochs // 10))) 
    
    plt.tight_layout()
    plt.show() # Use plt.show() if running locally, or plt.savefig('my_plot.png')
    # Use the function after your training is complete
    
plot_combined_history(history, history_ft)
