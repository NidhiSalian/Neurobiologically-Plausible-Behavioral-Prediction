# Neurobiologically-Plausible-Behavioral-Prediction
A model of mixed neural networks for step-by-step processing of dynamic visual scenes, activity recognition and behavioral prediciton

Based on Ongoing Research Work.

Code, datasets and models to be uploaded soon. 

## Pre-Requisites 

If you're working with Python for Deep Learning on Windows and would like to install Anaconda without any hassles, I'd recommend an older, stable release if the newer versions have unresolved bugs in their Windows versions. They can be found in the Anaconda Installer Archive [here](https://repo.continuum.io/archive/). 

Also recommended is the [Spyder IDE](https://anaconda.org/anaconda/spyder) that comes bundled with anaconda. 

_Note: If you're new to Python3 and Anaconda, [this](https://www.listendata.com/2017/05/python-data-science.html) might come in handy._

You'll need to install the following Python packages - links to conda installation commands provided : [OpenCV3](https://anaconda.org/conda-forge/opencv), [Keras](https://anaconda.org/conda-forge/keras)

(A common issue with libprotobuf in the openCV installation is resolved [here](https://github.com/ContinuumIO/anaconda-issues/issues/9601))

## Acknowledgements

I've used TimeDistributedDense() with my LSTM units to enable sequence to label mapping. It took me a while to wrap my head around it, but [this blogpost](https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/) and [this github issue thread](https://github.com/keras-team/keras/issues/1029) definitely helped.

## Note

I developed this particular project using Anaconda3 v4.4 for Windows, but I've made sure my code is 100% portable _as is_ to any other OS. I've already tested it with Linux variants. If you have any issues running this project on your setup despite meeting all the listed requirements, feel free to mail me at : nidhisalian08@gmail.com.

## License:

[GNU General Public License](./LICENSE)
