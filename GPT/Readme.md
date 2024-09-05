## Paper - [Attention is all you need](https://arxiv.org/abs/1706.03762)

while the paper had a encoder-decoder architecture with both cross-attention and self-attention this implementation is of decoder only transformer with self-attention.

The implementation is of a transformer based bigram model. 

## Demo 
[Demo Video on Linkedin](https://www.linkedin.com/feed/update/urn:li:activity:7237074573437923328/)

## Small parameters

The Hyper Parameters are on the smaller side due to lack of substantial GPU-resources to train a full-scale Transformer model.  
- batch size: 32, 
- context length : 8 , 
- no. of embedding : 32

## Dataset 
The implementation transformer.py can train the model on two datasets 
- tiny-shakespeare dataset which is all of Shakeshspeare in one file.
- quotes dataset which are bunch of quotes in a single txt file.

## Performance

The implementation was able to get the loss to 
- training loss : 1.984 and 
- validation loss : 2.091

