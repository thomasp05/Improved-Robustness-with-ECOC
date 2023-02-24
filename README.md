# Improved Robustness Against Adaptive Attacks with Ensembles and Error-Correcting Output Codes

## List of important files:

* **TrainEcocModel.py** : Train ECOC ensembles of shared and independent architecture on CIFAR10 and Fashion-MNIST. Also used to train ECOC ensembles with RegAdvT (adversarial examples used for adversarial training are generated on the whole ensemble and all binary classifiers are trained with the same perturbations)
* **AttackEcocModel.py** : Attack ECOC ensembles with FGSM, PGD, C&W_L2 attacks and all other adaptations presented in the paper.
* **TrainModel.py** : Train baseline models (SIMPLE and ENSEMBLE)
* **AttackModel.py** : Attack baseline models with FGSM, PGD, C&W_L2 attacks and all other adaptations presented in the paper.
* **IndAdvT.py** : Train ECOC ensembles with IndAdvT (adversarial examples used for adversarial training are generated on each binary classifier of the ECOC ensembles)
* **Models.py** : PyTorch architecture of ECOC and baseline models
* **generate_codewords.py** : script for generated random codewords. 

