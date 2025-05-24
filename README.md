# DRLF-DDI
We propose DRLF-DDI, a dual-view representation learning framework for DDIE prediction. Specifically, in the individual drug view, DRLF-DDI designs an autoencoder module with multi-head attention mechanism to generate enhanced Morgan fingerprints; in the interaction view, a DDI-Transformer module is introduced to capture the interaction features.

## Requirements

* Python == 3.7.3
* Pytorch == 1.8.1
* CUDNN == 11.1
* pandas == 1.3.0
* scikit-learn == 0.23.2
* rdkit-pypi == 2022.9.5
  
## Datasets
We use the same datasets as MRLF-DDI, and detailed descriptions are available at https://github.com/jianzhong123/MRLF-DDI.
  
## Files:
The source code files are in the ./codes folder. The details are as follows:
* construct_trans_graph_86.py: constructs the DDI event graph using the training dataset in the S0 setting (based on Ryu's dataset).
* construct_trans_graph.py: constructs the DDI event graph using the training dataset in the S0 setting (based on Deng's dataset).
* grid_cross5_model_S0_86.py: the model code in the S0 setting (based on Ryu's dataset).
* grid_cross5_model_S0_deng.py: the model code in the S0 setting (based on Deng's dataset).
  
## Running the code

The parameters are already set in the code files. You can run the following command to re-implement our work based on Ryu's dataset:

* > python Gridsearch_cross5_final_train_S0_86.py
* > python Gridsearch_cross5_final_train_S1_86.py
* > python  Gridsearch_cross5_final_train_S2_86.py
  
You can run the following command to re-implement our work based on Deng's dataset:
* > python Gridsearch_cross5_final_train_S0_Deng.py
* > python Gridsearch_cross5_final_train_S1_Deng.py
* > python Gridsearch_cross5_final_train_S2_Deng.py

## Contact

If you have any questions or suggestions with the code, please let us know. Contact Zhong Jian at jianzhong@csu.edu.cn
