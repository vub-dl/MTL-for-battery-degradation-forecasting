# MTL-for-battery-degradation-forecasting
This repository provides code for the experiments in: "**Towards Sustainable Power Systems: Exploring the Opportunities of Multi-task Learning for Battery Degradation Forecasting**": https://link.springer.com/chapter/10.1007/978-3-031-61069-1_9 


# Setup

Before running experiments, make sure the required dependencies are installed:

```
pip install -r requirements.txt 
```

To run the experiments on your own device, make sure to unpack all files (folders as well). 

The results of the experiments will be automatically logged to Weights and Biases https://wandb.ai/site. Make sure to create an account before starting the experiments.   

## Usage

To run benchmark experiments with different dynamic weighting algorithms on the "Battery Aachen dataset", run the **run_experiments.py** file. 

Before running the code, specify the configuration in the dictionary provided in the  **run_experiments.py** file. The meaning of each parameter is discussed below. 

