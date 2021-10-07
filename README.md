# Table of Contents  
[Introduction](#introduction)

[How to Run](#howtorun)  

[Resutls](#results)
<a name="introduction"/>
# Introduction
This repo presents code for Deep learning based algorithm for
**detecting violence** in indoor or outdoor environment.
The algorithm can detect following scenarios with high 
accuracy: fight, fire, car crash and even more.

To detect other scenarios you have to add **descriptive** text label of a 
scenario in `settings.yaml` file under `labels` key. At this moment model 
can detect 16`+1` scenarios, where one is default `Unknown` label. You can 
change, add or remove labels according to your use case. The model is trained on
wide variety of data and can generalize well on other scenarios too.
<a name="howtorun"/>    
# How to Run
To test the model you can either run:
`python run.py --image-path ./data/7.jpg`

Or you can see the example code in `tutorial.ipynb` jupyter notebook

Or you can 
<a name="results"/>
# Results
Here are resulting images