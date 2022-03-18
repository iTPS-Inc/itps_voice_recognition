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

For ease of usage, The Datasets are also bundled together in the `ENGLISH_DATASETS` and `JAPANESE_DATASETS` variables, making it possible to use the datasets like this.

```{python}
from dsets import get_datasets, DatasetConfig, JAPANESE_DATASETS
p, df = get_datasets(JAPANESE_DATASETS)
assert df["filename"].apply(os.path.exists).all()
assert len(df["original_dset"].unique()) == 2
assert set(df["original_dset"].unique()) == {"itps", "jsut"}
```

### Data
The `dsets` contains a helper python module for getting the datasets and getting them into an acceptable format.
There is a single function, `load_dset()`, with which it is possible to access all the datasets.

| Dataset           | usable? | train test split | language | password | Supported | Complete | Notes          | length |
|-------------------|---------|------------------|----------|----------|-----------|----------|----------------|--------|
| itps              | yes     | yes              | jp/en    | no       | yes       | yes      | -              | -      |
| librispeech       | yes     | yes              | en       | no       | yes       | yes      | -              | -      |
| LJ_Speech_Dataset | yes     | yes              | en       | no       | yes       | yes      | -              | -      |
| JSUT              | yes     | not yet          | jp       | no       | yes       | yes      | -              | -      |
| J-KAC             | no      | not yet             | jp       | yes      | x         | -        | -              | -      |
| NICT SPREDS       | no      | not yet          | ?        | ?        | x         | -        | -              | -      |
| RWCP              | no      | not yet          | ?        | ?        | x         | -        | -              | -      |
| PASD              | no      | not yet          | ?        | ?        | x         | -        | -              | -      |
| NICT_ASTREC       | no      | not yet          | ?        | ?        | x         | -        | -              | -      |
| JVS               | no      | not yet          | ?        | ?        | x         | -        | -              | -      |
| CENSREC-3         | no      | not yet          | ?        | ?        | x         | -        | -              | -      |
| CENSREC-1         | no      | not yet          | ?        | ?        | x         | -        | -              | -      |
| J-MAC             | no      | not yet          | jp       | yes      | x         | no       | No audio files | -      |

## Scripts
### Extracting data from excel files to make easy to use dataset.
