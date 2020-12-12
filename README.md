# MOLGNN_BASELINE

This code is implementation of baseline molecular graph neural networks with pytorch_geometric 

At first, one needs to convert raw data format into deep learning-ready. In the ./data folder, run preprocess.py
```python
python preprocess.py --dataset esol
```

We run training
```python
python train.py --dataset_file {dataset_path} --model {GCN, GAT, GIN} --save_model {path_to_save_your_trained_model} --save_result {path_to_save_your_classification_result} 
```
