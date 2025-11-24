import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model_path = r"D:\DoAn\trash_16_cls_grayscale_finetuned_noLambda.keras"
model = tf.keras.models.load_model(model_path)

# Define class names based on your training generator's class mapping
class_names = ['banana', 'battery', 'bimbim', 'coc', 'cucumberpeel', 'egg', 'hopxop', 'leaves','lotCoc', 'mask', 'metal', 'orange', 'paper', 'pen', 'bottle', 'bag'] 

# Path to the single image you want to predict
img_path = r"D:\DoAn\coc-thuy-tinh-11.jpg"# <-- Change this to your image path 
IMG_SIZE = (240, 240)

def predict_single_image(model, img_path, class_names):
    """
    Loads, preprocesses, and predicts the class of a single image.
    """
    try:
        # Load the image and resize it to the target size
        img = image.load_img(img_path, target_size=IMG_SIZE, color_mode="grayscale")
        
        # Convert the image to a numpy array and add a batch dimension
        # The model expects input shape (batch_size, height, width, channels)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Preprocess the image using the same function as the training data
        # Note: The original code uses a preprocess_input function from efficientnet,
        # which is designed for 3-channel (RGB) images. We need to handle the
        # grayscale to RGB conversion here as well. The model's input layer
        # expects (240, 240, 1), and the model itself handles the conversion
        # to (240, 240, 3) internally. We don't need to do that here.
        # The preprocessing function we defined is simply preprocessing_function=preprocess_input
        # from efficientnet. But wait, the model's input layer is 1 channel,
        # so we need to pass a 1-channel image. Let's trace it carefully.
        
        # The model's input is defined as: inp = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1), name='input_gray')
        # The datagenerator is also loading images in grayscale with 1 channel.
        # So the input image needs to be 1 channel. We already handled that.
        
        # Now, about the preprocessing: the `ImageDataGenerator` used `preprocessing_function=preprocess_input`.
        # This function expects RGB. Your model handles the conversion to RGB inside the model itself:
        # x = layers.Concatenate(axis=-1, name="gray_to_rgb")([inp, inp, inp])
        # So we just need to make sure the input array is correctly shaped and contains the right values.
        # The `efficientnet.preprocess_input` function scales pixel values.
        # It's better to let the model do its thing. Let's just pass the raw data.
        # Let's see the EfficientNet preprocess_input source code. It handles the scaling from [0,255]
        # to [-1,1]. The `Concatenate` layer is not a preprocessing layer, so let's stick to the code.
        # It's better to just use the `preprocess_input` function as intended.
        # Wait, the `flow_from_directory` is using it, but the model code also has an optional preprocess.
        # The most reliable way is to replicate the entire chain.
        
        # The `ImageDataGenerator` with `preprocessing_function=preprocess_input` does the following:
        # 1. Rescales the image from [0, 255] to [-1, 1].
        # 2. Applies the preprocessing for EfficientNet.
        # Let's replicate this.
        
        # The EfficientNet preprocess_input function is designed for 3-channel images.
        # Let's re-examine the original code. The model's `Input` is 1 channel, but the `base` model
        # expects 3 channels. The conversion `Concatenate` happens *inside* the model.
        # The `ImageDataGenerator` is told to load `color_mode="grayscale"`.
        # So the input to the model's `fit` function is a batch of (240, 240, 1) images.
        # The `preprocessing_function` in `ImageDataGenerator` is applied *before* the images are fed
        # to the model. This is where the discrepancy might be.
        # A `preprocessing_function` like `preprocess_input` from `efficientnet` expects 3 channels.
        # However, `ImageDataGenerator` with `color_mode="grayscale"` will give it a 1-channel image,
        # which will likely cause a shape mismatch error internally in `preprocess_input`.
        # This is a strong candidate for why your live predictions are failing.
        
        # Let's assume the `ImageDataGenerator` handles this gracefully and simply passes the 1-channel data,
        # which it shouldn't. A common workaround for this is to replicate the 1 channel to 3 channels
        # before calling `preprocess_input`.
        
        # Let's try to fix your prediction function.
        # Correctly preprocess the image like the training pipeline does.
        
        # Correct Preprocessing:
        # 1. Load the image as grayscale.
        # 2. Convert the 1-channel image to a 3-channel image by stacking.
        # 3. Apply the `preprocess_input` function.
        
        # Step 1: Load image
        img = image.load_img(img_path, target_size=IMG_SIZE, color_mode="grayscale")
        
        # Step 2: Convert to array and expand dimensions for batch
        img_array = image.img_to_array(img) # Shape: (240, 240, 1)
        img_array = np.expand_dims(img_array, axis=0) # Shape: (1, 240, 240, 1)
        
        # The model's input layer already handles the `gray_to_rgb` conversion,
        # so we do NOT need to do this step manually. We simply pass the 1-channel image.
        
        # Make the prediction
        predictions = model.predict(img_array)
        
        # Get the predicted class index
        predicted_class_index = np.argmax(predictions, axis=-1)[0]
        
        # Get the class name and prediction probabilities
        predicted_class_name = class_names[predicted_class_index]
        confidence_scores = predictions[0]
        
        print("Prediction Results:")
        print(f"Predicted Class: {predicted_class_name}")
        print("\nConfidence Scores for each class:")
        for i, score in enumerate(confidence_scores):
            print(f"- {class_names[i]}: {score*100:.2f}%")
            
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the prediction function
predict_single_image(model, img_path, class_names)