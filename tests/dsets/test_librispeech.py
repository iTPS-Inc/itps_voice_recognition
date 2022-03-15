import requests
from dsets.librispeech import LIBRISPEECH_DSETS

def test_datsets():
    for dset_name, url in LIBRISPEECH_DSETS:
        r = requests.get(dset_name)
        assert r.status_code == 200
