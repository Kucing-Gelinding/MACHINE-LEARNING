# CUNNY - AI Learning Navigator for Youth

## Documentation
- **Google Drive Folder**: [CUNNY Project Files](https://drive.google.com/drive/folders/1ZJyqlNGHTrLkvtDDl1ZWVSoC7Xss_o8e?usp=drive_link)
  - This folder contains all necessary resources and files required for the CUNNY project including datasets, trained models, and additional documentation.

### *Project Folder Structure*
For your project folder structure:
```
Engine/
│
├── Converter/  
│   └── Data/  
│       └── dcp.npz  
│   └── multiclass_npz.ipynb
│
├── Training/  
│   └── Data/  
│       ├── buah.h5  
│       ├── training_history.json  
│       └── training_history.csv  
│   └── Trainingmulti.ipynb
│
├── ForSave/  
│   └── DATA/  
│       └── fruitmaster.npz  
│   └── savefor_load.ipynb
│
└── Tester/  
    └── test pc/  
    ├── main.py  
    ├── image_classifier.py  
    ├── pict.jpg  
    ├── fruitmaster.npz  
    └── buah.h5
```
### *Requirements*
Ensure the following dependencies are installed:
- `numpy`
- `tensorflow`
- `matplotlib`
- Google Colab (if using)

Use this structured workflow to ensure smooth training and testing of your model.

## *How to Train the Model*

### *1. Prepare the Data*
- **Use the multiclass_npz script** to process and save your dataset as a `.npz` file.
- **Update the following paths**:
  - `DATA SAVE`: Directory to save the `.npz` file.
  - `DATA DIR`: Directory containing the input dataset.
- **If data augmentation is not required**, set `augmentation_model` to `None`.

### *2. Training the Model*

#### *Required File Paths*
- Update the following paths in the training script:
  - `npz_path`: Path to the `.npz` file generated from the data preparation step.
  - `model_path`: Path to save the trained model.
  - `csv_path`: Path to save the training history as a CSV file.
  - `json_path`: Path to save the training history as a JSON file.

#### *Data Preparation*
The script uses the `load_and_prepare_data` function to:
1. Load training and validation datasets from the `.npz` file.
2. Normalize image pixel values to the range [0, 1].
3. Shuffle and batch the datasets using TensorFlow’s `tf.data` API for efficient data loading.

#### *Model Architecture*
The model is defined with the following layers:
- Convolutional and MaxPooling layers to extract features.
- Fully connected layers with ReLU activation to learn representations.
- Dropout for regularization.
- Softmax activation for output classification.

#### *Training Configuration*
- **Loss function**: categorical_crossentropy
- **Optimizer**: RMSprop with a learning rate of 1e-4
- **Metrics**: Accuracy
- **Early stopping criteria**:
  - Training and validation accuracy reach 0.84.

#### *Training Process*
Run the training script with the following:
```bash
python run_train.py
```

- The trained model is saved as an `.h5` file at the specified `model_path`.
- Training history is saved to both JSON and CSV formats.

#### *Visualizing Training Progress*
The script plots training and validation accuracy/loss:
- Training curves are saved as a Matplotlib figure.

---

## *Model Testing and Prediction*

### *1. Run the Tester*
- Use the `savefor_load` script to prepare the test data:
  - `DATA SAVE`: Directory for saving the `.npz` file.
  - `DATA DIR`: Directory containing the dataset.
- Run the `main.py` script to execute tests.

### *2. Load and Predict*
- Use the `load_and_predict` function for predictions:
  - Input an image path to test the model.
  - The image is resized to `(200, 200)` to match the model's input size.
  - Predictions are displayed along with the input image.

Example:
```bash
python load_and_predict(model, '/content/example_image.jpg')
```

---

### *Visualization of Intermediate Layers*
The script includes functionality to visualize intermediate feature maps for debugging or model interpretability.
- Use the `visualize_intermediate_layers` function with:
  - Trained model.
  - Input image.
  - Optional rescaling layer.

Example:
```bash
img = load_image_from_npz(npz_path, image_index=10)
visualize_intermediate_layers(model, img, rescale_layer=tf.keras.layers.Rescaling(1./255))
```
---
