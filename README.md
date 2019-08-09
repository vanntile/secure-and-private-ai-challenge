# Secure and Private AI Scholarship Challenge
This is a repository with my work and notes from the Udacity ML&AI Challenge
sponsored by Facebook. I participated in this challenge in the summer of 2019.

The course is taught by [Mat Leonard](https://twitter.com/MatDrinksTea), the
head of Udacity's School of AI, and [Andrew Trask](http://iamtrask.github.io/),
the lead of OpenMined.org and researcher at DeepMind.

Two repositories that hold learning resources used in this course are:
- [Deep Learning with PyTorch](https://github.com/udacity/deep-learning-v2-pytorch)
- [Udacity's Private AI](https://github.com/udacity/private-ai)

## Content
- [lesson 2](./lesson-2/) concerns an intro to PyTorch
- [lessons 3 to 6](./lesson-3-6) is on differential privacy and PATE analysis
- [lesson-7](./lesson-7) is on federated learning
- [lesson-8](./lesson-8) is on securing federated learning
- [lesson-9](./lesson-9) is on encrypting deep learning
- [PATE MNIST Analysis](./PATE MNIST Analysis.ipynb) is a proposed project
on using general differential privacy to train an unlabaled dataset

## PATE MNIST analysis

This project concerns using general differential privacy to train an unlabeled dataset using predictions from interface-compatible models.

Task: Assume we have a private unlabaled dataset and access to  external models trained to similar datasets and with a common interface.  We want to train our model on our dataset using the predictions of the  others for labeling, while keeping the external models's data and our  own within some privacy parameters.

Here we simulate the whole process, from the training of the external  datasets to our own and measuring the epsilon-privacy using PATE.

Frameworks used in this project:
- pytorch
- numpy
- matplotlib
- pysift

Following, there are some code snippets highlighting the method:

```python
# Detecting and using CUDA for training, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
images, labels = images.to(device), labels.to(device)
```

```python
# Simulating several separated initial datasets by splitting them equally
indices = list(range(dataset_size))

np.random.seed(random_seed)
np.random.shuffle(indices)

indices = list(split(indices, no_datasets))

# Creating data samplers and loaders:
samplers = list()
for i in indices:
    sampler = SubsetRandomSampler(i)
    samplers.append(sampler)

train_dataloaders = list()
for s in samplers:
    loader = torch.utils.data.DataLoader(dataset_train, batch_size=32, sampler=s)
    train_dataloaders.append(loader)
```

```python
# Training a model
model = Network(784, 10, [512, 256, 128])
optimizer = optim.Adam(model.parameters(), lr=0.001) # low learning rate to simulate real life
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
    
acc = train(model, train_dataloaders[i], dataloader_test, criterion, optimizer, epochs=1)
```

And training sample output:

```
Epoch: 1/1.. 
Training Loss: 1.9363..  Test Loss: 1.0918..  Test Accuracy: 0.6603
Training Loss: 1.0398..  Test Loss: 0.6290..  Test Accuracy: 0.7999
Training Loss: 0.7456..  Test Loss: 0.4959..  Test Accuracy: 0.8483
Training Loss: 0.6424..  Test Loss: 0.4478..  Test Accuracy: 0.8605
Training Loss: 0.6003..  Test Loss: 0.3816..  Test Accuracy: 0.8854
```

```python
# Analysing data dependent and independent epsilon using PATE
data_dep_eps, data_ind_eps = pate.perform_analysis(teacher_preds=np.asarray(predictions), indices=consensus, noise_eps=eps, delta=1e-5, moments=4)
```

