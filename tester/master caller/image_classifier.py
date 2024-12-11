# image_classifier.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class ImageClassifier:
    def __init__(self, model):
        self.model = model

    def load_and_predict(self, image_path):
        img = load_img(image_path, target_size=(200, 200)) 
        img_array = img_to_array(img)
        img_array = img_array.reshape((1,) + img_array.shape)
        img_array = img_array / 255.0

        # Make a prediction
        predictions = self.model.predict(img_array)

        # Get the predicted class
        predicted_class = np.argmax(predictions[0])
        print("Predicted Class:", predicted_class)

        return predicted_class, img_array[0]

    def show_single_image_from_npz(self, npz_file, index=0, save_path=None):
        data = np.load(npz_file)

        images = data['images']
        labels = data['labels']
        names = data['names']

        image = images[index]
        label = labels[index]

        index = np.argmax(label)

        plt.imshow(image.astype("uint8"))
        plt.title(f"Label: {names[index]}")
        plt.axis("off")

        if save_path:
            plt.savefig(save_path, format='jpg', bbox_inches='tight')
            print(f"Image saved as: {save_path}")
        plt.close()

    def run(self, pc_path, npz_file_path, savepc):
        try:
            predicted_class, _ = self.load_and_predict(pc_path)

            self.show_single_image_from_npz(npz_file_path, predicted_class, savepc)

            return 1
        except Exception as e:
            print(f"An error occurred: {e}")

            return 0
