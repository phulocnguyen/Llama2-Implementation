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

More detail in file pdf ASM

#Submission

Report: your zip file can include a pdf file, named Group_ID-report.pdf. Report can present implement the requirements, accuracy, best results are with some hyperparameters other than the default, how running your code... no more than 3 pages in 2 weeks.
Canvas Submission
For submission via Canvas, the submission file should be a zip file with the following structure:
GROUP_ID/

├── run_llama.py

├── base_llama.py

├── llama.py

├── rope.py

├── classifier.py

├── config.py

├── optimizer.py

├── sanity_check.py

├── tokenizer.py

├── utils.py

├── README.md

├── structure.md

├── sanity_check.data

├── generated-sentence-temp-0.txt

├── generated-sentence-temp-1.txt

├── [OPTIONAL] sst-dev-advanced-output.txt

├── [OPTIONAL] sst-test-advanced-output.txt

├── sst-dev-prompting-output.txt

├── sst-test-prompting-output.txt

├── sst-dev-finetuning-output.txt

├── sst-test-finetuning-output.txt

├── [OPTIONAL] cfimdb-dev-advanced-output.txt

├── [OPTIONAL] cfimdb-test-advanced-output.txt

├── cfimdb-dev-prompting-output.txt

├── cfimdb-test-prompting-output.txt

├── cfimdb-dev-finetuning-output.txt

├── cfimdb-test-finetuning-output.txt

├── Group_ID-report.pdf

└── setup.sh

