# ASSIGNMENT 1: 
# DEVELOPMENT MINIMALIST OF THE LLAMA2 MODEL (core components of Llama2)

TO DO IN THIS ASSIGNMENT:
- Implement some important components of the Llama2 model to better understanding its architecture. 
- Perform sentence classification on sst dataset and cfimdb dataset with this model.

DETAIL:
The code to implement can be found in llama.py, classifier.py and optimizer.py. You are reponsible for writing core components of Llama2 (one of the leading open source language models). 
In doing so, you will gain a strong understanding of neural language modeling. We will load pretrained weights for your language model from stories42M.pt; an 8-layer, 42M parameter language model pretrained on the TinyStories dataset (a dataset of machine-generated children's stories). Download tại link:  https://www.cs.cmu.edu/~vijayv/stories42M.pt
This model is small enough that it can be trained (slowly) without a GPU. You are encouraged to use Colab or a personal GPU machine (e.g. a Macbook) to be able to iterate more quickly.

Once you have implemented these components, you will test our your model in 3 settings:
1.	Generate a text completion (starting with the sentence "I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is"). You should see coherent, grammatical English being generated (though the content and topicality of the completion may be absurd, since this LM was pretrained exclusively on children's stories).
2.	Perform zero-shot, prompt-based sentiment analysis on two datasets (SST-5 and CFIMDB). This will give bad results (roughly equal to choosing a random target class).
3.	Perform task-specific finetuning of your Llama2 model, after implementing a classification head in classifier.py. This will give much stronger classification results.

More detail in file ASM (ASSIGNMENT 1 DEVELOPMENT MINIMALIST OF THE LLAMA2 MODEL.pdf)
