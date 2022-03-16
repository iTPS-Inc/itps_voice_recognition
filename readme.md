# Transcription

## Dataset module
The dataset module `dsets` exists to make access to the dataset we have at iTPS easy.
It takes care of automatically downloading the data, caching that download, and getting the 
data into the correct shape for instant usage.

### Usage
The main interface to the `dsets` module is the `get_datasets` function and the `DatasetConfig` object.
The `get_datasets` takes a list of `DatasetConfig` objects and builds a dataset from these.

Below is an example for usage of this API to receive both and both splits of the itps dataset, and the
`dev` split form the `other` dataset of `librispeech`.

The `get_datasets` function returns a list of paths, with the roots to each of the dataset and a 
`pd.Dataframe` with information about all of the requested data. The two columns `text` and `filename`
are guaranteed to be in the dataset.

```{python}
from dsets import get_datasets, DatasetConfig

p, df = get_datasets(
    [
        DatasetConfig(name="itps", lang="both", split="both"),
        DatasetConfig(name="librispeech", kind="other", split="dev"),
    ]
)
```

The `DatasetConfig` only required argument for the DatasetConfig is the `name` argument.
If `split` is not passed, it defaults to the `train` datset.


### Data
The `dsets` contains a helper python module for getting the datasets and getting them into an acceptable format.
There is a single function, `load_dset()`, with which it is possible to access all the datasets.

| Dataset           | usable? | language | password | Supported | Complete | Notes          | length |
|-------------------|---------|----------|----------|-----------|----------|----------------|--------|
| itps              | yes     | jp/en    | no       | yes       | yes      | -              | -      |
| librispeech       | yes     | en       | no       | yes       | yes      | -              | -      |
| LJ_Speech_Dataset | yes     | en       | no       | yes       | yes      | -              | -      |
| JSUT              | yes     | jp       | no       | yes       | yes      | -              | -      |
| J-KAC             | no      | jp       | yes      | x         | -        | -              | -      |
| NICT SPREDS       | no      | ?        | ?        | x         | -        | -              | -      |
| RWCP              | no      | ?        | ?        | x         | -        | -              | -      |
| PASD              | no      | ?        | ?        | x         | -        | -              | -      |
| NICT_ASTREC       | no      | ?        | ?        | x         | -        | -              | -      |
| JVS               | no      | ?        | ?        | x         | -        | -              | -      |
| CENSREC-3         | no      | ?        | ?        | x         | -        | -              | -      |
| CENSREC-1         | no      | ?        | ?        | x         | -        | -              | -      |
| J-MAC             | no      | jp       | yes      | x         | no       | No audio files | -      |

## Scripts
### Extracting data from excel files to make easy to use dataset.
