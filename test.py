import sys
import numpy as np
import os
import joblib


def lubohong_test(img_path, model_path, scaler_path):
    def load_mnist(img_path, kind='train'):
        """
        'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'
        Before use, you need to download the above four files to the `path` directory and unzip them
        """
        labels_path = os.path.join(img_path, '%s-labels.idx1-ubyte' % kind)
        images_path = os.path.join(img_path, '%s-images.idx3-ubyte' % kind)

        with open(labels_path, 'rb') as label_file:
            labels = np.frombuffer(label_file.read(), dtype=np.uint8, offset=8)

        with open(images_path, 'rb') as image_file:
            images = np.frombuffer(image_file.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 784)

        return images, labels

    x_test, y_test = load_mnist(img_path, kind='t10k')

    # Loading saved models
    svm_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    x_test = scaler.transform(x_test)

    # predict
    y_predict = svm_model.predict(x_test)

    # Output results
    print('Predicted values:\n', y_predict)
    print('True Values:\n', y_test)
    print('Direct comparison of true and predicted values：\n', y_test == y_predict)
    print('Accuracy：\n', svm_model.score(x_test, y_test))


if __name__ == "__main__":

    arg_num = len(sys.argv)
    model_nums = ('1', '2', '3', '4', '5')

    if arg_num == 2:
        number = str(sys.argv[1])
        if number in model_nums:
            print("Start testing!")
            lubohong_test(
                img_path="./data/",
                model_path="./models/svm_model%s.pkl" % number,
                scaler_path="./models/transfer%s.pkl" % number
            )
            print("Testing completed!")
        elif number == 'all':
            for i in model_nums:
                print('Start testing model svm_model%s' % i)
                lubohong_test(
                    img_path="./data/",
                    model_path="./models/svm_model%s.pkl" % i,
                    scaler_path="./models/transfer%s.pkl" % i
                )
                print('Model svm_model%s testing completed!' % i)
        else:
            print('Parameter values do not conform to the format')
            print("Desirable range of parameters: ", model_nums, " or run \"python test.py all\" to run all models")
            print("For example, the command to test the first model is: ", "\"python test.py 1\"")
    else:
        print('The number of parameters does not conform to the format')
        print("Desirable range of parameters: ", model_nums, " or run \"python test.py all\" to run all models")
        print("For example, the command to test the first model is: ", "\"python test.py 1\"")

