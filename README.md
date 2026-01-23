# README

# UAGTF

## Install

Make sure your local environment has the following installed:

```undefined
python == 3.8.10
pytorch == 2.2.2(GPU version) + cuda == 12.1
transformers == 4.46.3
numpy == 1.24.2
pandas == 2.0.3
sklearn == 1.3.2
scipy == 1.10.1
tqdm
```

## Run

```undefined
python main_{dataset}.py
```
You can change the parameters in the `main_{dataset}.py`​ file to control the training process, and the final results are saved in folder `/trained_models/{dataset}/{model_name}`​.

## Test

```undefined
python main_{dataset}.py --only_test
```

You must use ` --only_test`​ to choose mode

You must place the `model.pt`​ file in the `/trained_models/{dataset}/checkpoint`​ directory.
