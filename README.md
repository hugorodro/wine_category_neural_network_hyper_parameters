I.	Single Hidden Layer MLP

a.	Preparing the Data: 

The data was split on two occasions. The first, between training (80% of total) and testing (20% of total). The second, to create a new training set (90% of previous training) and validation set (10% of previous training). The feature columns were normalized between zero and one.

a.	Network Design:

 Of course, a single hidden layer was used. I settled with a size of 12 nodes. I came to this number by using PCA to find number of features that would give me a variance >= 99%. The activation function used is ReLU. I could have used sigmoid since it is a shallow network, but performance was similar while I was developing the model. Below is the output of the model from the terminal.

NeuralNetwork(
  (hiddenLayers): Linear(in_features=13, out_features=12, bias=True)
  (layers): Sequential(
    (0): Linear(in_features=13, out_features=12, bias=True)
    (1): ReLU()
    (2): Linear(in_features=12, out_features=3, bias=True)
  )
)

b.	Training Parameters

To achieve the results in the following sub sections, I trained the model with the parameters below:
•	Batch size of 32 - Provided a good balance between stable training and jumping out of local minima. 
•	Cross Entropy Loss Function
•	ADAM optimizer function - This helped with rapid convergence compared to SGD) 
•	Stopping Criteria where the number of occurrences of a condition (next Validation Loss > the previous Validation Loss) was equal to 100.
•	A learning rate of .001 – Experimentally, this had a good balance convergence speed and validation performance.

c.	Results:

Metrics
Epochs	227
Training Loss	0.14
Validation Loss	0.30
Validation Accuracy	0.86
Test Accuracy 	0.97

Validation Confusion Matrix
2	2	0
0	4	0
0	0	6

Test Confusion Matrix
13	1	0
0	14	0
0	0	7


B.	Effect of Performance on a Variety of Parameters

The goal was to see how different hyper parameters affected the performance of the neural networks. Below there are 4 tables that demonstrate this. The first table shows the categories (network design, learning rate, and stopping criteria). The second, third, and fourth table shows how the average value of the metrics when the parameter value was set (green). The best performing parameter is also show (yellow). 

Experimentation Parameters
Hidden Layers & Nodes	12	12, 6	12, 8, 4
Learning Rate	0.0001	0.001	0.01
Stopping Criteria	Training Loss	Validation Loss	Validation Accuracy

Effect of Hidden Layers
Hidden Layers & Nodes	12	12, 6	12, 8, 4
Average # of Epochs	183.44	187.67	185.56
Average Training Loss	0.59	0.32	0.23
Average Validation Loss	0.53	0.32	0.33
Average Validation Accuracy	0.88	0.86	0.86
Average Test Accuracy	0.9	0.84	0.85


Effect of Learning Rate
Learning Rate	0.0001	0.001	0.01
Average # of Epochs	180.44	179.22	184.11
Average Training Loss	0.05	0.04	0.28
Average Validation Loss	0.11	0.19	0.29
Average Validation Accuracy	0.97	0.97	0.9
Average Test Accuracy	0.97	0.98	0.9


Effect of Stopping Criteria
Stopping Criteria	Training Loss	Validation Loss	Validation Accuracy
Average # of Epochs	216.56	214	117.11
Average Training Loss	0.1	0.05	0.03
Average Validation Loss	0.1	0.21	0.05
Average Validation Accuracy	0.97	0.93	0.98
Average Test Accuracy	0.96	0.97	0.98


























