import sys
import numpy as np
import os
import joblib


def lubohong_test(img_path, model_path, scaler_path):
    def load_mnist(img_path, kind='train'):
        """
        'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'
        使用前需要把上面四个文件下载到 `path` 目录下并解压
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

    # 加载保存的模型
    svm_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    x_test = scaler.transform(x_test)

    # 预测
    y_predict = svm_model.predict(x_test)

    # 输出结果
    print('预测值:\n', y_predict)
    print('真实值:\n', y_test)
    print('直接比对真实值和预测值：\n', y_test == y_predict)
    print('准确率：\n', svm_model.score(x_test, y_test))


if __name__ == "__main__":

    arg_num = len(sys.argv)
    model_nums = ('1', '2', '3')

    if arg_num == 2:
        number = str(sys.argv[1])
        if number in model_nums:
            print("开始测试！")
            lubohong_test(
                img_path="./data/",
                model_path="./models/svm_model%s.pkl" % number,
                scaler_path="./models/transfer%s.pkl" % number
            )
            print("测试完成！")
        else:
            print('参数数值不符合格式')
            print("参数可取范围：", model_nums)
            print("例如，测试第一个模型的命令为：", "python test.py 1")
    else:
        print('参数个数不符合格式')
        print("参数可取范围：", model_nums)
        print("例如，测试第一个模型的命令为：", "python test.py 1")

