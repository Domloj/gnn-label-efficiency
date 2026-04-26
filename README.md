# GNN Label Efficiency - Semi-Supervised Node Classification

A study on how the number of available labels affects the performance 
of Graph Neural Network architectures in semi-supervised node classification 
on citation graphs.

## Research Question
How does label availability (5%, 10%, 20%) impact the accuracy 
of GCN, GAT, and GraphSAGE on the Cora citation graph dataset?

## Key Contribution
Unlike original papers (Kipf & Welling 2017, Veličković 2018, Hamilton 2017) 
which evaluated models on a fixed benchmark split of 140 labeled nodes, 
this project systematically compares all three architectures across 
controlled label ratio scenarios.

## Models
- **GCN** - Graph Convolutional Network (Kipf & Welling, 2017)
- **GAT** - Graph Attention Network (Veličković et al., 2018)
- **GraphSAGE** - Inductive Representation Learning (Hamilton et al., 2017)

## Dataset
- **Cora** - 2,708 nodes, 5,429 edges, 7 classes
- **Citeseer**

## Experiments
| Model | 5% | 10% | 20% |
|-------|----|-----|-----|
| GCN   | 0.7922 ± 0.0147  | 0.8152 ± 0.0143   | 0.8274 ± 0.0147   |
| GAT   | -  | -   | -   |
| GraphSAGE | - | - | -  | -   |

*Results to be filled in during experiments.*

## References
- Kipf & Welling (2017) - Semi-Supervised Classification with GCN
- Veličković et al. (2018) - Graph Attention Networks
- Hamilton et al. (2017) - Inductive Representation Learning on Large Graphs
