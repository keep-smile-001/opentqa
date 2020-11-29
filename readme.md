# OPENTQA

OPENTQA is a open framework of the textbook question answering.   

##  Structure

Our method can be generalized by this picture:

![框架图](https://raw.githubusercontent.com/keep-smile-001/opentqa/master/pic.png)

##	Virtual Environment Installing

You can simply install the virtual environment by:

```     
pip install -r requirements.txt
```

##  Datasets 

The preprocessed dataset is placed in

```  https://drive.google.com/drive/folders/1eFx3XGyUSs0ij12wSmv70e6MRXa2TEFL
 https://drive.google.com/drive/folders/1eFx3XGyUSs0ij12wSmv70e6MRXa2TEFL
```

The datasets were split to 2 parts, diagram questions (dqa) and non-diagram questions (ndqa) ; 

For each model, the project provides argumentations below: 

+ dqa

  > + que_ix: the index of questions
  > + opt_ix: the index of options
  > + dia: the diagram corresponding the above question
  > + ins_dia: the instructional diagram corresponding to the lesson that contains the above question
  > + cp_ix: the closest paragraph that is extracted by TF-IDF method

+ ndqa

  > + que_ix: the index of questions
  > + opt_ix: the index of options
  > + cp_ix: the closest paragraph that is extracted by TF-IDF method

##  Models

OPENTQA contains several models which are listed down here.

> + xtqa
> + mcan
> + mfb
> + CMR
> + mutan

Each model is composed by four parts:

> + <tt> **[model_name].yml** </tt> This file is in <tt>configs/[dataset]/</tt> and the other 3 files are in <tt>opentqa/models/[dataset]/</tt>.It contains the global variables of the model, such as <tt>data_path</tt> , <tt>epoch</tt> , <tt>batch_size</tt> etc.
> + **<tt>net.py</tt>**  This file describes how the model works.
> + <tt>**model_cfgs.py**</tt> This file contains the local variables of the model.
> + **<tt>layers.py</tt>** This file contains layers which might be called in <tt>net.py</tt>

You can also add **your own model** to the project by following these steps:

>1. Add **<tt>net.py</tt>**  to <tt>opentqa/models/[dataset]/</tt>  , it should accept the argumentation mentioned above;
>2. Add **<tt>model_cfgs.py</tt>**  and **<tt>layers.py</tt>** according to your need;
>3. Add <tt> **[model_name].yml** </tt>  to <tt>configs/[dataset]/</tt> , make sure it contains the global variables and the correct path of dataset.

Next we will talk about how to run the code.

##  How to run the code?

> Before you try to run the code, please make sure your configs are right:<tt>configs/[dataset]/[model_name].yml</tt> 

You can train the model by the command:

```python
 python run.py --dataset_use=[dataset] --model=[model_name] --run_mode=train 
```

For example, if I want to train "xtqa" on dataset "dqa", the command should be:

``` 
 python run.py --dataset_use=dqa --model=xtqa --run_mode=train 
```

You can find your a model version showed on the terminal, and this number is used to test the model.

The result will be in <tt>results/log/</tt> and the checkpoints will be placed in <tt>ckpt/</tt>

> Notice: if there are two same version, the new one will cover the old one!

You can test the model by the command:

``` 
python run.py --ckpt_v=[ckpt_version] --ckpt_e=[ckpt_epoch] --dataset_use=[dataset] --model=[model_name] --run_mode=test
```

<tt>ckpt_version</tt> was talked above, <tt>ckpt_epoch</tt> refers to the epoch of the checkpoint.

For example, you can test the 5th epoch of "xtqa" which version is 1024 on dqa on this way:

``` 
python run.py --ckpt_v=1024 --ckpt_e=5 --dataset_use=dqa --model=xtqa --run_mode=test
```

