# Transcription
## Data
The `dsets` contains a helper python module for getting the datasets and getting them into an acceptable format.
There is a single function, `load_dset()`, with which it is possible to access all the datasets.

| Dataset           | password | language | Supported | Complete | Notes          |
|-------------------|----------|----------|-----------|----------|----------------|
| itps-corpus       | no       | jp/en    | yes       | yes      | -              |
| librispeech       | no       | en       | yes       | yes      | -              |
| J-MAC             | yes      | jp       | x         | x        | No audio files |
| J-KAC             | yes      | jp       | x         | -        | -              |
| LJ_Speech_Dataset | ?        | ?        | x         | -        | -              |
| JSUT              | ?        | ?        | x         | -        | -              |
| NICT SPREDS       | ?        | ?        | x         | -        | -              |
| RWCP              | ?        | ?        | x         | -        | -              |
| PASD              | ?        | ?        | x         | -        | -              |
| NICT_ASTREC       | ?        | ?        | x         | -        | -              |
| JVS               | ?        | ?        | x         | -        | -              |
| CENSREC-3         | ?        | ?        | x         | -        | -              |
| CENSREC-1         | ?        | ?        | x         | -        | -              |

## Scripts
### Extracting data from excel files to make easy to use dataset.
