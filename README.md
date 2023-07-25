# Graph-Neural-Network-Fortification-with-Adversarial-Learning-

## Introduction
In this project, we use adversarial learning methods to attack our GNN model that was trained for malware detection and family classification. We study the performance of GNNs against adversarial examples, and we improve its robustness by retraining it with successful attacks.

## Graph Neural Networks (GNN)
Graph Neural Networks are deep learning methods based on iterative neighborhood aggregation. These models learn feature vector representations of nodes and graph (sub)structures in a lower dimensional space. GNNs have demonstrated state-of-the-art performance in many tasks, including node, edge, and graph classification [\[1\]](https://arxiv.org/pdf/1810.00826.pdf). That is why we use GNNs in our research.
We train our GNN model using malware behavior graphs to generate d-dimensional vectors representing each graph. These vectors  can be considered a dynamic signature of malware.
For the implementation, we use the popular [GraphGym](https://github.com/snap-stanford/GraphGym) library by Standford Machine Learning Group.

## Adversarial Learning
Most machine learning and deep learning models, including GNNs, are vulnerable to adversarial examples. An adversarial example is an input sample that was slightly modified with the intention of fooling the machine learning classifier [\[2\]](https://arxiv.org/pdf/1804.00097.pdf).
In many domain applications, including image classification, the changes are very small and may not be perceived by the naked eye. However, for malware detection, the generated adversarial samples need to maintain the functionalities of the original samples [\[3\]](https://nrl.northumbria.ac.uk/id/eprint/49453/1/Accepted%20Manuscript.pdf).

## Preliminary Results

Performance of GNN on original malware: Accuracy 0.918, F1_score 0.957

Performance of GNN on adversarial malware: Accuracy 0.644, F1_score 0.783

Performance of retrained GNN on adversarial malware: Accuracy 0.976, F1_score 0.976
