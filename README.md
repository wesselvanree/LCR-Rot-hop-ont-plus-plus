# LCR-Rot-hop-ont++

Source code for Injecting Knowledge from a Domain Sentiment Ontology in a Neural Approach for Aspect-Based Sentiment
Classification.

> [!NOTE]
> Looking for the initial release (2023)? View the [initial release branch](https://github.com/wesselvanree/LCR-Rot-hop-ont-plus-plus/tree/initial-release).

## Getting started

### Data

First, create a `data/raw` directory and download
the [SemEval 2015](http://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools), [SemEval 2016](http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools)
datasets, and the [ontology](https://github.com/KSchouten/Heracles/tree/master/src/main/resources/externalData). Then
rename the SemEval datasets to end up with the following files:

- `data/raw`
  - `ABSA15_Restaurants_Test.xml`
  - `ABSA15_Restaurants_Train.xml`
  - `ABSA16_Restaurants_Test.xml`
  - `ABSA16_Restaurants_Train.xml`
  - `ontology.owl-Extended.owl`

### Installing packages in a virtual environment

This project uses `uv` to manage its dependencies. [Install uv](https://docs.astral.sh/uv/getting-started/installation/) on your local machine. Clone this repository to your local machine, and open a terminal window in this repository. Create a virtual environment using:

```
uv venv --python 3.11
```

Then, install packages using

```
uv sync
```

### Running scripts

All entrypoints are located in the `src/lcr/entrypoints` directory. Each entrypoint accepts CLI arguments. To view the available cli args for a program, run `python [ENTRYPOINT] --help`. For example, you can use CLI args to pick the year of the dataset.

- `preprocess.py`: remove opinions that contain implicit targets and generate embeddings, these embeddings are used
  by the other programs. To generate all embeddings for a given year, run `python -m lcr.entrypoints.preprocess --all`
- `hyperparam.py`: run hyperparameter optimization
- `train.py`: train the model for a given set of hyperparameters
- `validate.py`: validate a trained model. To do an ablation experiment, run `python -m lcr.entrypoints.validate --ablation`,
  this requires all embeddings to be created for a given year.

Thus, you might execute these commands:

```shell
python -m lcr.entrypoints.preprocess --year 2015 --all
python -m lcr.entrypoints.preprocess --year 2015 --phase Train --ont-hops 1
python -m lcr.entrypoints.hyperparam --year 2015
python -m lcr.entrypoints.hyperparam --year 2015 --val-ont-hops 1
# The best params for each configuration can be found in the results directory, please edit the training process accordingly
python -m lcr.entrypoints.train --year 2015 # Add ont-hops arguments if you want
python -m lcr.entrypoints.validate --year 2015 --model TRAINED_MODEL_PATH
python -m lcr.entrypoints.validate --year 2015 --model TRAINED_MODEL_PATH --ablation
```

## Acknowledgements

The `lcr.model.bert_encoder` module uses code from:

- Liu, W., Zhou, P., Zhao, Z., Wang, Z., Ju, Q., Deng, H., Wang, P.: K-BERT: Enabling language representation with
  knowledge graph. In: 34th AAAI Conference on Artificial Intelligence. vol. 34, pp. 2901–2908. AAAI Press (2020)
- https://github.com/Felix0161/KnowledgeEnhancedABSA
