This is the python version of captcha recognition implemented by MXNet. It is modified from [reference](https://github.com/xlvector/learning-dl/blob/master/mxnet/ocr/cnn_ocr.py), but the main difference is this version can recognize a captcha composed of number or lowercase letters or capital. For example:

![](captcha_example.png)

### Usage

clone this project and just run:

`python captcha_train.py`

### Attention

* The result of model is sensitive to `learning rate` 
