# Transcription
## Data
The `dsets` contains a helper python module for getting the datasets and getting them into an acceptable format.
There is a single function, `load_dset()`, with which it is possible to access all the datasets.

| Dataset           | password | language | Supported | Complete | Notes          | length |
|-------------------|----------|----------|-----------|----------|----------------|--------|
| itps-corpus       | no       | jp/en    | yes       | yes      | -              | -      |
| librispeech       | no       | en       | yes       | yes      | -              | -      |
| J-MAC             | yes      | jp       | x         | no       | No audio files | -      |
| J-KAC             | yes      | jp       | x         | -        | -              | -      |
| LJ_Speech_Dataset | no       | en       | yes       | yes      | -              | -      |
| JSUT              | no       | jp       | yes       | yes      | -              | -      |
| NICT SPREDS       | ?        | ?        | x         | -        | -              | -      |
| RWCP              | ?        | ?        | x         | -        | -              | -      |
| PASD              | ?        | ?        | x         | -        | -              | -      |
| NICT_ASTREC       | ?        | ?        | x         | -        | -              | -      |
| JVS               | ?        | ?        | x         | -        | -              | -      |
| CENSREC-3         | ?        | ?        | x         | -        | -              | -      |
| CENSREC-1         | ?        | ?        | x         | -        | -              | -      |

## Scripts
### Extracting data from excel files to make easy to use dataset.
