# DDSC: Masked Discrete Diffusion for Gene Expression 

## Motivation
Gene expression count vectors are tricky to navigate with machine learning models: they have a very peaked distribution around small values but it is also long tailed. 
Moreover, the count distributions can be cell dependent and higher variable per sample. 
This makes it challenging to train a model to predict count vectors. When predicting raw counts and moreover predicting the mean (say under L2 loss), without hand crafted loss function, the model will be **dominated by the large counts** and will not be robust to errors of small counts. 

For these reasons, it may be natural to instead consider gene expression count *ranks*. Moreover, we recognize that ranks have a natural categorical definition, as we define up to $n$ ranks or buckets. 

With this motivation, we explore framing the problem with a "sequence to sequence" lens. Additionally, we frame it as a generative modeling problem: given some sequence of some count ranks, where some count rank tokens are missing ([MASK]), we aim to predict or inpaint the "masked" count ranks. 

This fits well with a masked language discrete diffusion model, which we train on the Tahoe 100M dataset. 

### Method 
We train a conditional diffusion model on the gene expression count rank sequences. Specifically, we condition on the gene IDs. We follow the MosiacFM work and consider sequences of length 1024. 
We train using a DiT backbone. 

Currently, we have train for 70k batches or approximately 4.5 million cells. 
With limited training we get around 0.6 NLL
<p align="center">
  <img src="./training_loss.png" alt="Training Loss Curve" width="500">
</p>

### Inpainting Generation
We can do flexible controllable generation with our discrete diffusion model and toggle the level and location of masking. 
![Diffusion Animation](./diffusion.gif)





### Code acknowledgements
This repo builds on the MosaicFM repo (specifically using their dataloaders). It adapts MDLM code from https://github.com/kuleshov-group/mdlm/.

                                                              |

