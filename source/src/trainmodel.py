import argparse
import os
import sys
import numpy as np
from deepface import DeepFace
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

def main(args):
    # Load dataset
    dataset = load_dataset(args.data_dir)
    
    # Split dataset if needed
    if args.use_split_dataset:
        train_set, test_set = split_dataset(dataset, args.min_nrof_images_per_class, args.nrof_train_images_per_class)
    else:
        train_set = dataset
        test_set = load_dataset(args.test_data_dir) if args.test_data_dir else dataset

    # Get image paths and labels
    train_paths, train_labels = get_image_paths_and_labels(train_set)
    test_paths, test_labels = get_image_paths_and_labels(test_set)

    if args.mode == 'TRAIN':
        # Train the model
        print('Training model...')
        model = train_model(train_paths, train_labels)
        
        # Save the model
        with open(args.classifier_filename, 'wb') as file:
            pickle.dump(model, file)
        print(f'Saved classifier model to file "{args.classifier_filename}"')

    elif args.mode == 'CLASSIFY':
        # Load the model
        with open(args.classifier_filename, 'rb') as file:
            model = pickle.load(file)
        print(f'Loaded classifier model from file "{args.classifier_filename}"')
        
        # Classify test images
        print('Classifying test images...')
        predictions = classify_images(model, test_paths)
        
        # Calculate and print accuracy
        accuracy = accuracy_score(test_labels, predictions)
        print(f'Accuracy: {accuracy:.3f}')

def load_dataset(data_dir):
    dataset = []
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            image_paths = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
            dataset.append(ImageClass(class_name, image_paths))
    return dataset

def get_image_paths_and_labels(dataset):
    image_paths = []
    labels = []
    for cls in dataset:
        image_paths.extend(cls.image_paths)
        labels.extend([cls.name] * len(cls.image_paths))
    return image_paths, labels

def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        if len(cls.image_paths) >= min_nrof_images_per_class:
            np.random.shuffle(cls.image_paths)
            train_set.append(ImageClass(cls.name, cls.image_paths[:nrof_train_images_per_class]))
            test_set.append(ImageClass(cls.name, cls.image_paths[nrof_train_images_per_class:]))
    return train_set, test_set

def train_model(image_paths, labels):
    embeddings = DeepFace.represent(img_path=image_paths, model_name=args.model, enforce_detection=False)
    model = DeepFace.build_model(args.model)
    model.fit(embeddings, labels)
    return model

def classify_images(model, image_paths):
    embeddings = DeepFace.represent(img_path=image_paths, model_name=args.model, enforce_detection=False)
    predictions = model.predict(embeddings)
    return predictions

class ImageClass:
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'], help='Indicates if a new classifier should be trained or an existing one should be used for classification', default='CLASSIFY')
    parser.add_argument('data_dir', type=str, help='Path to the data directory containing aligned face images.')
    parser.add_argument('model', type=str, help='Model name for DeepFace (e.g., VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, ArcFace).')
    parser.add_argument('classifier_filename', type=str, help='Classifier model file name as a pickle (.pkl) file. For training this is the output and for classification this is an input.')
    parser.add_argument('--use_split_dataset', action='store_true', help='Indicates that the dataset specified by data_dir should be split into a training and test set.')
    parser.add_argument('--test_data_dir', type=str, help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--min_nrof_images_per_class', type=int, help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int, help='Use this number of images from each class for training and the rest for testing', default=10)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
