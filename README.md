### This is the python version of captcha recognition implemented by MXNet. 

It is modified from [reference1:xlvector](https://github.com/xlvector/learning-dl/blob/master/mxnet/ocr/cnn_ocr.py). There is also a R version of captcha recognition of MXNet [reference2:incubator-mxnet](https://github.com/apache/incubator-mxnet/tree/master/example/captcha). **However the main difference is this version can recognize a captcha composed of number or lowercase letters or capital**. For example:

![](captcha_example.png)

### Usage

* If you want to train from scratch, please run:

`python captcha_train_from_scratch.py`

* If you want to fine tune in the pretrained model(**recommend**), please run:

`python captcha_train_finetune.py`


### Attention
* pretrained model can be download from:[GoogleDrive](https://drive.google.com/open?id=1-yEReei1jD3sUS5tzg5uNVwKUJwy6r9W)

* If you train from scratch, the result of model is sensitive to `learning rate` and is a little difficult to converge, so **fine tune is recommended**. 
