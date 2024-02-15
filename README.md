# Bridge Microstrain Prediction

## Overview

This project implements a system which uses a dataset containing a set of truck images passing over a bridge, and their corresponding sensor data which has recorded various features such as mid-span strains, temperature, etc. over more than 3000 samples, and tries to predict the number of axles of each truck based on these informations. As of results, the model achieved 52.4% and 62.2% accuracy on train and validation sets respectively. The model consists of three major components, which are:


1. A CNN to embed the pictures' tensors
2. An LSTM to embed the sensors' data tensors
3. A Stacked Attention Network (SAN) to fuse the two embeddings to improve performance by leveraging attention layers' mechanism.

<br>

## Key features 
### Size: 
Comprising 100  images  from cameras and 100 strain sensor samples set up on a controlled bridge.
### Privacy Assurance: 
The dataset creation process prioritizes the anonymity and confidentiality of confidential information, addressing privacy concerns.


## Requirements

To run the code in the Jupyter Notebook, you'll need the following Python libraries:

- `PIL`
- `numpy`
- `pandas`
- `nltk`
- `os`
- `tqdm`
- `matplotlib`
- `torch`
- `torchtext`
- `torchvision`

You can install these dependencies using the following command:

```bash
pip install pillow numpy pandas nltk tqdm matplotlib torch torchtext torchvision
```

Additionally, make sure you have Jupyter Notebook installed:
```bash
!pip install jupyter
 ```

## Training

To train the model, use the provided `train_model` function in your Jupyter Notebook. The function takes the following parameters:

- `model`: The neural network model you want to train.
- `criterion`: The loss function, in this case, it's `nn.CrossEntropyLoss`.
- `optimizer`: The optimization algorithm, in this case, it's Adam.
- `num_epochs`: The number of epochs for training.

Here's an example of how to use the function:
1. **Load the previous cells :**

```python
# Import necessary libraries
# Define your model, criterion, and optimizer
model = TruckModel(input_representation_shape=512, embedd_dim=EMBEDD_DIM, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters())

model_trained = train_model(model, criterion, optimizer, num_epochs=10)
```
## Evaluation

To evaluate the trained model on a test set and calculate the accuracy, precision, recall, and F1-score, use the following function which takes a model, and a dictionary which contains the data loaders in it:

1. **Load Trained Model:**

```python
# Import necessary libraries
# Set the model to evaluation mode
def evaluate(model, dataloaders):

    model.eval()
    
    prec_rec_f1 = {'train': [[], [], []], 'validation': [[], [], []]}
    acc = {'train': [], 'validation': []}

    for phase in ['train', 'validation']:
        
        with tqdm(dataloaders[phase], unit='batch', position=0, leave=True) as pbar:
            
            for X, y, z in pbar: 
                X = X.to(device)
                y = y.to(device).to(torch.float32)
                z = z.to(device)
                
                outputs = model(X, y)
                _, preds = (torch.max(outputs, 1))
                
                batch_prec_rec_f1_score = precision_recall_fscore_support(np.array(z.cpu()), np.array(preds.cpu()), average='micro')
                batch_acc = accuracy_score(np.array(z.cpu()), np.array(preds.cpu()))
                
                prec_rec_f1[phase][0].append(batch_prec_rec_f1_score[0])
                prec_rec_f1[phase][1].append(batch_prec_rec_f1_score[1])
                prec_rec_f1[phase][2].append(batch_prec_rec_f1_score[2])
                acc[phase].append(batch_acc)
                
                pbar.set_postfix(prec=batch_prec_rec_f1_score[0],
                                            rec=batch_prec_rec_f1_score[1],
                                            f1=batch_prec_rec_f1_score[2],
                                            acc=batch_acc)


        prec_rec_f1[phase][0] = np.mean(np.array(prec_rec_f1[phase][0]))
        prec_rec_f1[phase][1] = np.mean(np.array(prec_rec_f1[phase][1]))
        prec_rec_f1[phase][2] = np.mean(np.array(prec_rec_f1[phase][2]))
        acc[phase] = np.mean(np.array(acc[phase]))


    print(f'Precision on train set: {prec_rec_f1["train"][0]}')
    print(f'Precision on validation set: {prec_rec_f1["validation"][0]}')
    print()
    print(f'Recall on train set: {prec_rec_f1["train"][1]}')
    print(f'Recall on validation set: {prec_rec_f1["validation"][1]}')
    print()
    print(f'F1-Score on train set: {prec_rec_f1["train"][2]}')
    print(f'F1-Score on validation set: {prec_rec_f1["validation"][2]}')
    print()
    print(f'Accuracy on train set: {acc["train"]}')
    print(f'Accuracy on validation set: {acc["validation"]}')
```










