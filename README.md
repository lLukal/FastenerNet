# FastenerNet

### Description

Deep learning models for classification of fasteners into 6 categories and 142 subcategories.
Created for Bachelor's degree project and thesis for University of Zagreb Faculty of Electrical Engineering and Computing.
Worked on mainly by Luka Miličević. Initial project steps and first simple models created together with Fran Kufrin and Dorjan Štrbac.

### Usage

Docker and docker compose required (it's recommended to use docker engine). In the starting directory run:
```
sudo docker compose up --build
```
For correct usage the fastener dataset is required, placed in the folder fastener_dataset.

### Components

Three types of jupyter notebook:
- create_dataset (create training, validation and test datasets in different resolutions, grayscale, with and without 'Misc' category)
- model_6 and model_142 (train models for 6 and 142 class classification tasks)
- inference_6 and inference_142 (evaluate models for 6 and 142 class classification tasks)

Trained models (models directory), model training and evaluation results (model_results directory), created datasets (datasets directory).
