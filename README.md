# Singular-Value-Decomposition-CV

Image Classification with self-implemented logistic regression while utilizing Singular Value Decomposition.

A report could be found [here](doc/report.pdf).


## Description

In this project, we used NumPy implemented Logistic Regression with Stochastic Gradient Descent to classify Japanese Hiragana hand writings from the KMNIST dataset. We then used Singular Value Decomposition to reduce the size of images for the goals of decreasing memory allocations and hopefully increasing the performance of the model. After applying Singular Value Decomposition, we were able to achieve 99% of testing accuracy on classifying お and ま with 40% less memory allocation on the original images as well as around 87% of testing accuracy on classifying す and ま with 40% less memory allocation on the original images.


## Getting Started

### Dependencies

* Python3
* `Numpy`
* `Matplotlib`

### Installs

* `NotoSansJP-Regular.otf` (Japanese fonts).
    * You may download the entire family [here](https://fonts.google.com/noto/specimen/Noto+Sans+JP), or you may just save the `NotoSansJP-Regular.otf` in this repo.


### Files
* `data` folder contains all of our datasets.
* `out_images` folder contains all of our generated plots. The plots will be updated once you re-run the program.
* `NotoSansJP-Regular.otf` is a Japanese font collection that you need to have in this directory so that the program will execute normally.
* `data.py` implements all given methods to load, normalize, shuffle, and generate minibatches without any extra add-ons.
* `image.py` saves the sample data as pictures.
* `image_playground.ipynb` visualizes the sample images for each model.
* `network.py` implements our neural network for Logistic Regression.
* `main.py` is the only script you need to run. The execution instructions are specified below. By running this program, you will be able to see the testing accuracy of all three sets of data and get newly generated plots in the `out_images` folder.

### Executing the Program

In your terminal, run the following command to execute the program:

```python
python3 main.py
```

For other arguments, use the argument parser defined in `main.py`.

## Help

If the Japanese Hiragana characters aren't shown on the `weights.png`, please make sure you have `NotoSansJP-Regular.otf` in your directory.

