# Neurobiologically-Plausible-Behavioral-Prediction
A model of mixed neural networks for step-by-step processing of dynamic visual scenes, activity recognition and behavioral prediciton

Based on Ongoing Research Work.

Code, datasets and models to be uploaded soon. 

## Pre-Requisites 

If you're working with Python for Deep Learning on Windows and would like to install Anaconda without any hassles, I'd recommend an older, stable release if the newer versions have unresolved bugs in their Windows versions. They can be found in the Anaconda Installer Archive [here](https://repo.continuum.io/archive/). 

Also recommended is the [Spyder IDE](https://anaconda.org/anaconda/spyder) that comes bundled with anaconda. 

_Note: If you're new to Python3 and Anaconda, [this](https://www.listendata.com/2017/05/python-data-science.html) might come in handy._
<br />

You'll need to install the following Python packages :
 <details>
 <summary>Keras</summary>
 This line should let you [install Keras]((https://anaconda.org/conda-forge/keras)) (I used v2.1.5): 
 
 `conda install -c conda-forge keras`
 </details>
 <details>
 <summary>OpenCV</summary>
 This line should let you [install OpenCV](https://anaconda.org/conda-forge/opencv) (I used v3.4.1): 
 
 `conda install -c conda-forge opencv`
 
 ( _Linux Users- a common issue with OpenCV - resolved [here](https://github.com/conda-forge/opencv-feedstock/issues/43)_)
 </details>
<br />

>If you want to train your own model, you'll need a GPU. A massive ammount of processing power is required to learn the numerous parameters(69,163,810 - last I checked) in the hidden dimensions of the LSTM units. If you already have a GPU on your system, Keras should be able to detect it automatically. (Nvidia users can run a quick `nvidia-smi ` to confirm. )

## Acknowledgements

I've used TimeDistributed() wrappers with my LSTM units to enable sequence to label mapping. It took me a while to wrap my head around it, but [this blogpost](https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/), [this github issue thread](https://github.com/keras-team/keras/issues/1029) and [this StackOverflow thread](https://stackoverflow.com/questions/46859712/confused-about-how-to-implement-time-distributed-lstm-lstm) definitely helped.

## Note

I developed this particular project using Anaconda3 v4.4 for Windows, but I've tried to make sure my code is 100% portable _as is_ to any other OS. I've already tested it with Linux variants. If you have any issues running this project on your setup despite meeting all the listed requirements, feel free to mail me at : nidhisalian08@gmail.com.

## License:

[GNU General Public License](./LICENSE)
