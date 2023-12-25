***This repository is used to display the work on a project to train a neural network to determine the capillaries of the eye from photographs.***
=
**To copy the work you need to create an anaconda environment with command _conda env create -n <your_env_name> spec.yaml_ and maintain the original structer**
-
*__EyeDataset.py__ the file contains the EyeDataset class, which is the definition of the dataset class. Also you can find the wrapper above the EyeDataset - __PartDataset__ class, the purpose of which is to split the data.* \
*File __metrics.py__ contains the definition of metrics that are used in training NN training.* \
*File __UnetTrainer.py__ contains class UnetTrainer, which is the main class to train the NN.* \
*File __main.py__ is using to train the Unet NN on augmented data. In this file you can find commented code sections, which is the alternative data augmentation.* \
*File __test.py__ is using to evaulate trained NN on both validate and test datasets* 

> ## STRUCTURE:
>
> PP
> > model (this folder is not in repo, but it should be)
> >
> > Dataset
> > > eye_test
> > > 
> > > eye_train
> >
> > EyeDataset.py
> > 
> > main.py
> > 
> > metrics.py
> > 
> > test.py
> > 
> > UnetTrainer.py
> > 
> > spec.yaml
> >

*Markups for dataset pictures is in geojson format with polygon/multipolygon types, investigate the structure of this to create your own dataset* \
*Using Unet NN from __https://github.com/qubvel/segmentation_models.pytorch/tree/master__. Check out this repo!* \
*Pretrained weights is stored here:*
* [Default augmentation](https://drive.google.com/file/d/1xZ5nHI_BWo7VlI9UH8X6FmqItIQWxNWi/view?usp=sharing)
* [Semi additional augmentation](https://drive.google.com/file/d/1j_Jc7UsmFrXQNGiw1ndEBpW9xPJ52h4e/view?usp=sharing)
* [Excessive additional augmentation](https://drive.google.com/file/d/1SjYPfzN2MeycDhyQvImkkqKmO7yAF_oE/view?usp=sharing) 

#### *You should store weights into model folder* 
\
\
\
\
***Этот репозиторий служит для отображения работы над проектом по обучению нейросети для определения капилляров глаза по фотографиям.***
=
**Чтобы скопировать результат работы, вам нужно создать среду anaconda с помощью команды _conda env create -n <your_env_name> spec.yaml_**
-
*EyeDataset.py файл содержит класс EyeDataset, который является определением класса dataset`a. Также вы можете найти обёртку над классом EyeDataset - PartDataset, целью которой является разделение данных* \
*Файл __metrics.py__ содержит определение показателей, которые используются при обучении НС.* \
*Файл __UnetTrainer.py__ содержит класс UnetTrainer, который является основным классом для обучения НС.* \
*Файл __main.py__ используется для обучения Unet на дополненных данных. В этом файле вы можете найти прокомментированные разделы кода, которые являются альтернативным дополнением данных.* \
*Файл __test.py__ используется для оценки обученной НС как для валидационного, так и для тестового наборов данных*
> ## Структура:
>
> PP
> > model (этой папки нет в проекте, но она должна быть)
> >
> > Dataset
> > > eye_test
> > > 
> > > eye_train
> >
> > EyeDataset.py
> > 
> > main.py
> > 
> > metrics.py
> > 
> > test.py
> > 
> > UnetTrainer.py
> > 
> > spec.yaml
> >

*Разметки для изображений набора данных представлены в формате geojson с типами polygon/multipolygon, изучите структуру этого, чтобы создать свой собственный набор данных* \
*Используя Unet NN из __https://github.com/qubvel/segmentation_models.pytorch/tree/master __. Ознакомьтесь с этим репозиторием!* \
*Предварительно подготовленные веса хранятся здесь:*
* [Стандартные данные](https://drive.google.com/file/d/1xZ5nHI_BWo7VlI9UH8X6FmqItIQWxNWi/view?usp=sharing)
* [Данные, подвергшиеся небольшой аугментации](https://drive.google.com/file/d/1j_Jc7UsmFrXQNGiw1ndEBpW9xPJ52h4e/view?usp=sharing)
* [Чрезмерная аугментация данных](https://drive.google.com/file/d/1SjYPfzN2MeycDhyQvImkkqKmO7yAF_oE/view?usp=sharing)\
*Вам следует сохранить веса в папке model* 
