* ECA automation - v0.1

* Base language model : Flan-T5-small 

* ECA functions: 
    - generate reponse with a prefix "respose:" 
    - classify topics with a prefix "topic: "

* Execution: 

    $ python train_model_FT5.py -model=uichat -batch=16 -lr="0.0003" -epoch=100


The code includes:

1. Preprocessing (simple):
     1) combine the new data to the existing data
     2) stratified random split the given data into "training" and "validation" 
     
2. Training: 
     1) Hyperprameter search (only stopped epoch this time)  
         - train the model with the training data
         - early stop with the validation data (get an optimal training epoch)
         
     2) Final model training
       - train all the data, using that optimal epoch
       - test all the data
     
3. Evaluation:
    1) topic classification (Currently only available evaluation)
    2) response generation 
         - might need to evaluate with simiarlity metric in the future
         

    ** Evaluation results (only topic classification):
    topic_res: 1719 / 3580
    * Topic classification:
       topic       mcc        f1       acc  
    0  macro  0.936321  0.745042  0.942408 
         


4. Development environment
    * OS: CentOS Linux 7 (Core)
    * miniconda install: https://docs.conda.io/projects/conda/en/latest/user-guide/install/
    * packages (see the installation guide at the bottom):
        python=3.10 
        numpy=1.25.2         
        pytorch=2.0.1
        transformers=4.23.1
        sentence-transformers=2.2.2
        datasets=2.10.1
        nvidia-ml-py3=7.352.0
        nltk=3.8
        scikit-learn=1.2
                

4. Installation guide with conda:
    1) Linux
        $ conda create -n nlp python=3.10 numpy=1.25
        $ conda install pytorch=2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia
             # torch gpu check: $ python -c "import torch; print(torch.cuda.is_available())"
        $ conda install -c conda-forge accelerate              # >= 0.20.1 ( Using the `Trainer` with `PyTorch` requires `accelerate>=0.20.1`)
        $ conda install -c conda-forge sentence-transformers
        $ conda install -c anaconda scikit-learn
        $ pip install transformers datasets nvidia-ml-py3
