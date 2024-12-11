from image_classifier import ImageClassifier
import tensorflow as tf

def main():
    
    model = tf.keras.models.load_model(r'C:\Users\acer\Downloads\cp final capter\tester\master caller\buah.h5')  

    pc_path = r'C:\Users\acer\Downloads\cp final capter\tester\master caller\test pc\Screenshot 2024-11-21 190943.png'  
    npz_file_path = r'C:\Users\acer\Downloads\cp final capter\tester\master caller\fruitmaster(1).npz'  
    savepc = r'C:\Users\acer\Downloads\cp final capter\tester\master caller\pict.jpg'  

    classifier = ImageClassifier(model)

    result = classifier.run(pc_path, npz_file_path, savepc)

    if result == 1:
        print("RUN")
    else:
        print("FAIL")

if __name__ == "__main__":
    main()
