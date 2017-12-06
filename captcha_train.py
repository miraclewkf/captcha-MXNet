import mxnet as mx
import numpy as np
import cv2, random
import os
import argparse
from captcha.image import ImageCaptcha

def get_captcha(length, captcha_str):
    result = ""
    for i in range(length):
        item = random.randint(0,len(captcha_str)-1)
        captcha_item = captcha_str[item]
        result += captcha_item
    return result

def get_label(num, captcha_dic):
    result = []
    for item in num:
        result.append(int(captcha_dic[item]))
    return result

class CustomDataIter(mx.io.DataIter):
    def __init__(self, num_example, batch_size, label_length, height, width, captcha_str, captcha_dic):
        super(CustomDataIter, self).__init__()
        self.captcha = ImageCaptcha(fonts=['./fonts/Ubuntu-M.ttf'])
        self.batch_size = batch_size
        self.num_example = num_example
        self.label_length = label_length
        self.height = height
        self.width = width
        self.captcha_str = captcha_str
        self.captcha_dic = captcha_dic
        self.provide_data = [('data', (batch_size, 3, height, width))]
        self.provide_label = [('softmax_label', (self.batch_size, label_length))]

    def __iter__(self):
        for k in range(self.num_example / self.batch_size):
            data = []
            label = []
            for i in range(self.batch_size):
                num = get_captcha(length = self.label_length, captcha_str = self.captcha_str)
                img = self.captcha.generate(num)
                img = np.fromstring(img.getvalue(), dtype='uint8')
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (self.width, self.height))
                cv2.imwrite("./tmp" + str(i % 10) + ".png", img)
                img = np.multiply(img, 1/255.0)
                img = img.transpose(2, 0, 1)
                data.append(img)
                label.append(get_label(num, self.captcha_dic))

            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(label)]
            data_batch = mx.io.DataBatch(data = data_all, label = label_all)
            yield data_batch

    def reset(self):
        pass

def get_net(num_class):
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('softmax_label')
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=32)
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(2,2), stride=(1, 1))
    relu1 = mx.symbol.Activation(data=pool1, act_type="relu")

    conv2 = mx.symbol.Convolution(data=relu1, kernel=(5,5), num_filter=32)
    pool2 = mx.symbol.Pooling(data=conv2, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu2 = mx.symbol.Activation(data=pool2, act_type="relu")

    conv3 = mx.symbol.Convolution(data=relu2, kernel=(3,3), num_filter=32)
    pool3 = mx.symbol.Pooling(data=conv3, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu3 = mx.symbol.Activation(data=pool3, act_type="relu")

    flatten = mx.symbol.Flatten(data = relu3)
    fc1 = mx.symbol.FullyConnected(data = flatten, num_hidden = 512)
    fc21 = mx.symbol.FullyConnected(data = fc1, num_hidden = num_class)
    fc22 = mx.symbol.FullyConnected(data = fc1, num_hidden = num_class)
    fc23 = mx.symbol.FullyConnected(data=fc1, num_hidden = num_class)
    fc24 = mx.symbol.FullyConnected(data=fc1, num_hidden = num_class)
    fc2 = mx.symbol.Concat(*[fc21, fc22, fc23, fc24], dim = 0)
    label = mx.symbol.transpose(data = label)
    label = mx.symbol.reshape(data = label, shape = (-1, ))
    return mx.symbol.SoftmaxOutput(data = fc2, label = label, name = "softmax")

# define the accuracy class for captcha, for example: if ground truth is "R23k", only you predict "R23k" is right
class Accuracy_captcha(mx.metric.EvalMetric):
    def __init__(self, batch_size):
        super(Accuracy_captcha, self).__init__('accuracy_captcha')
        self.batch_size = batch_size

    def update(self, labels, preds):
        pred = preds[0].asnumpy()
        label = labels[0].asnumpy().astype('int32')

        label = label.T.reshape((-1,))
        hit = 0
        total = 0
        for i in range(pred.shape[0] / 4):
            ok = True
            for j in range(4):
                k = self.batch_size * j + i
                if int(np.argmax(pred[k])) != int(label[k]):
                    ok = False
                    break
            if ok:
                hit += 1
            total += 1

        self.sum_metric += hit
        self.num_inst += total

def multi_factor_scheduler(begin_epoch, epoch_size, step=[20,40], factor=0.2):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None

if __name__ == '__main__':

    # hyper parameter define
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--train-data-number', type=int, default=100000, help="number of captcha sample for train")
    parser.add_argument('--test-data-number', type=int, default=5000, help="number of captcha sample for test")
    parser.add_argument('--num-epoch', type=int, default=50, help="number of train epoch")
    parser.add_argument('--height', type=int, default=30, help="the height of input image")
    parser.add_argument('--width', type=int, default=80, help="the width of input image")
    parser.add_argument('--label-length', type=int, default=4, help="the length of captcha")
    parser.add_argument('--output-path', type=str, default='output/test', help="the path to save model")
    args = parser.parse_args()

    # the item of the captcha
    captcha_str = "0123456789abcdefghijklmn"

    # define the int label from a to Z using dictionary
    captcha_dic = {}
    label_begin = 0
    for i in range(0, len(captcha_str)):
        captcha_dic[captcha_str[i]] = label_begin
        label_begin += 1

    # get network
    network = get_net(num_class = len(captcha_str))

    # define gpu
    devs = [mx.gpu(0)]

    optimizer_params = {
                'learning_rate': 0.0002,
                'momentum' : 0.9,
                'wd' : 0.00001}

    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)

    # create a model based on network you get
    model = mx.mod.Module(
            context       = devs,
            symbol        = network
        )

    # get train and test data
    data_train = CustomDataIter(num_example = args.train_data_number,
                                batch_size = args.batch_size,
                                label_length = 4,
                                height = args.height,
                                width = args.width,
                                captcha_str = captcha_str,
                                captcha_dic = captcha_dic)
    data_test = CustomDataIter(num_example = args.test_data_number,
                               batch_size = args.batch_size,
                               label_length = 4,
                               height = args.height,
                               width = args.width,
                               captcha_str = captcha_str,
                               captcha_dic = captcha_dic)

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    # define learning rate decay policy
    epoch_size = max(int(args.train_data_number / args.batch_size), 1)
    lr_scheduler = multi_factor_scheduler(0, epoch_size)

    # define evaluation metric
    eval_metric = mx.metric.CompositeEvalMetric()
    eval_metric.add(Accuracy_captcha(batch_size = args.batch_size))
    eval_metric.add(['CrossEntropy'])

    # define the output file path of model
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # train model
    model.fit(data_train,
              eval_data = data_test,
              eval_metric = eval_metric,
              validation_metric= eval_metric,
              num_epoch = args.num_epoch,
              optimizer_params = optimizer_params,
              initializer = initializer,
              batch_end_callback=mx.callback.Speedometer(args.batch_size, 50),
              epoch_end_callback=mx.callback.do_checkpoint(args.output_path))