# Deep-PICO

Experiments in deep (OK, shallow, but using embeddings) for PICO identification.

##Requirements

python2.7

Keras
```bash
    $ pip install keras
```

scikit-learn
```bash
    $ pip install -U scikit-learn
```
gensim
```bash
    $ pip install gensim
```
theano
```bash
    $ pip install theano
```
nltk
```bash
    $ pip install nltk
```
geniatagger
```bash
    got to http://www.nactem.ac.uk/GENIA/tagger/
    unzip: tar xvzf geniatagger.tar.gz
    navigate to geniatagger and make
    
    install the python wrapper
    pip install geniatagger-python
```
sklearn_crfsuite
```bash
    $ pip install theano
```
pycrfsuite
```bash
    $ pip install theano
```


Installing tensorflow
```bash
    # Ubuntu/Linux 64-bit
    $ sudo apt-get install python-pip python-dev
    
    # Mac OS X
    $ sudo easy_install pip
    # Ubuntu/Linux 64-bit, CPU only:
    $ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl
    
    # Ubuntu/Linux 64-bit, GPU enabled:
    $ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl
    
    # Mac OS X, CPU only:
    $ sudo easy_install --upgrade six
    $ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.7.1-cp27-none-any.whl
```

##Usage

###Running the Conditional Random Field Model
```bash
    $ python crf.py 
```

###Command line arguments 
```bash
    --w2v       # 1 or 0 whether to use word vectors or not as features
    --iters     # number of iterations to train on    
    --l1        # l1 regulerzation term
    --l2        # l2 regulerzation term
    --wiki      # 1 or 0 whether to use the word vectors trained on wikipedia and pubmed 
    --shallow_parse     # 1 or 0 whether to use standerd POS features
    --words_before      # number of words to use as features that come before each token
    --words_after       # number of words to use as features that come after each token
    --grid_search       # 1 or 0 whether to search for optimal hyperparmeters with grid search
```




###Running the Convolutional or Standard Neural Network 
To use the  Convolutional Neural Network or Standard Feed forward Neural Network

```bash
    $ python GroupCNNExperiment.py  
```

###Command line arguments

```bash
    --window_size       # the number of words to use as features 
    --wiki              # 1 or 0 Use the word vectors trained on pubmed and wikipedia 
    --n_feature_maps    # the numner of feature maps for the CNN only
    --epochs            # number of epochs to train the model for
    --undersample       # 1 or 0 whether to train the model with 
    --criterion         # the loss function
    --optimizer         # optimization algorthim 
    --model             # nn or cnn | whether to use a Convotuonal or feed forward neural network 
    --genia             # 1 or 0
    --tacc              # for personal use only or if you have access to TACC for some reason 
    --layers            # format <1,2,3,4> the numbers of hidden layers in the network
```
