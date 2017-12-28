# Transfer Learning - Dogs-vs-Cats-Redux
This repository uses pretrained model [Vgg-16](http://arxiv.org/abs/1409.1556.pdf) to fine tune over a dataset of Dogs-vs-Cats-Redux challenge. 
The checkpoint file can be downloaded from [here](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz).

## Directory Structure
```
Project
|-- datasets
|   |-- dev_set
|   |-- test
|   |-- test_set
|   |-- train
|   `-- train_set
|-- model
|-- pretrained-model
|-- submissions
|-- datalab.py
|-- dataset_clusterer.py
|-- make_file.py
|-- model.py
|-- vgg16.py
|-- predict.py
|-- test.py
`-- train.py
```

## Kaggle Score
This approach gave me a log loss score of about: 0.08426