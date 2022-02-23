# %% 
import torch
import flash 
# %%
from flash.audio import SpeechRecognition, SpeechRecognitionData
from flash.core.data.utils import download_data
# %%
download_data("https://pl-flash-data.s3.amazonaws.com/timit_data.zip", "./data")

from ipywidgets import Audio
# %%
datamodule = SpeechRecognitionData.from_json(
    input_fields="file",
    target_fields="text",
    train_file="data/timit/train.json",
    test_file="data/timit/test.json",
)
# %%

model = SpeechRecognition(backbone="facebook/wav2vec2-base-960h")
# %%
trainer = flash.Trainer(max_epochs=1, gpus=torch.cuda.device_count())
trainer.finetune(model,
    datamodule=datamodule, strategy="no_freeze")

# %%
predictions = model.predict(["data/timit/example.wav"])
# %%
print(predictions)

# %%
trainer.save_checkpoint("speech_recognition_model.pth")


# %%
