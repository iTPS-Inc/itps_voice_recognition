# Transcription
## Data
The `dsets` contains a helper python module for getting the datasets and getting them into an acceptable format.
There is a single function, `load_dset()`, with which it is possible to access all the datasets.

| Dataset           | usable? | language | password | Supported | Complete | Notes          | length |
|-------------------|---------|----------|----------|-----------|----------|----------------|--------|
| itps-corpus-jp    | yes     | jp/en    | no       | yes       | yes      | -              | -      |
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
