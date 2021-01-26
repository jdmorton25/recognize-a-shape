# Recognize a shape
The program can recognize 3 types of shape: circle, rectangle, triangle.\
Firstly, you need load or train your model. For training model you need create a dataset. You can generate it if you run 'data.py' (in file fields 'train_samples_count', 'test_samples_count', this is amount of pictures which will be generated). So you prepare to train your model, run 'main.py' and press 'train' (it is reaaaaly long process).\
Program has a canvas where you can draw a shape and press 'predict'. If you need draw new picture, so press 'clear canvas' to clear that. There is a prediction in the left bottom corner.\
Also there are options, you can change brush color and brush width.\

* main.py - main program (program runs a bit long because of import tensorflow I think and I don't know how to fix that)\
* data.py - data generator (there are some settings picture size, train/test pictures amont)\
* model.h5 - file of saved model\
* data.zip - dataset which I generated\

Screenshot of program interface:\
![1](https://user-images.githubusercontent.com/9623983/105824541-ed5f8680-5fd7-11eb-8d58-86b05929591e.png)
