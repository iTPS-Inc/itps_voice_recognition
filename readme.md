# Transcription
## Data
The `dsets` contains a helper python module for getting the datasets and getting them into an acceptable format.
There is a single function, `load_dset()`, with which it is possible to access all the datasets.

| Dataset           | password | language | Supported | Complete | Notes          |
|-------------------|----------|----------|-----------|----------|----------------|
| J-MAC             | yes      | JP       | x         | x        | No Audio files |
| J-KAC             | yes      | JP       | x         | -        | -              |
| LJ_Speech_Dataset | ?        | ?        | x         | -        | -              |
| JSUT              | ?        | ?        | x         | -        | -              |
| NICT SPREDS       | ?        | ?        | x         | -        | -              |
| RWCP              | ?        | ?        | x         | -        | -              |
| PASD              | ?        | ?        | x         | -        | -              |
| NICT_ASTREC       | ?        | ?        | x         | -        | -              |
| JVS               | ?        | ?        | x         | -        | -              |
| CENSREC-3         | ?        | ?        | x         | -        | -              |
| CENSREC-1         | ?        | ?        | x         | -        | -              |
| itps-corpus       | x        | jp/en    | Yes       | Yes      | -              |
| librispeech       | x        | en       | Yes       | Yes      | -              |

## Scripts
### Extracting data from excel files to make easy to use dataset.
