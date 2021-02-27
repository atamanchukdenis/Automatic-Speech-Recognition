import os
import numpy as np
import tensorflow as tf
import automatic_speech_recognition as asr

#running only on GTX 1080 until RTX 2060 is fixed
os.environ["CUDA_VISIBLE_DEVICES"]="1"

dataset = asr.dataset.Audio.from_csv('/home/datamanc/data/CommonVoice/en/cv-corpus-6.1-2020-12-11/en/train_dataset.csv', batch_size=64)
dev_dataset = asr.dataset.Audio.from_csv('/home/datamanc/data/CommonVoice/en/cv-corpus-6.1-2020-12-11/en/val_dataset.csv', batch_size=64)
alphabet = asr.text.Alphabet(lang='en')
lm = asr.text.LanguageModel('/home/datamanc/data/CommonCrawl/400K_3-gram.binary').load()
features_extractor = asr.features.FilterBanks(
    features_num=160,
    winlen=0.02,
    winstep=0.01,
    winfunc=np.hanning
)
model = asr.model.get_deepspeech(
    input_dim=160,
    output_dim=28
)
optimizer = tf.optimizers.Adam(
    lr=1e-4,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8
)
decoder = asr.decoder.GreedyDecoder()
#decoder = asr.decoder.PrefixBeamSearchDecoder(
#    lm=lm)
spec_augment = asr.augmentation.SpecAugment(
    F=40,
    mf=1,
    Tmin=10,
    Tmax=30,
    mt=5
)
pipeline = asr.pipeline.CTCPipeline(
    alphabet, features_extractor, model, optimizer, decoder,
    checkpoint_dir='checkpoint'
)
pipeline.fit(dataset, dev_dataset, epochs=2, augmentation=spec_augment)
pipeline.save('checkpoint')

test_dataset = asr.dataset.Audio.from_csv('/home/datamanc/data/CommonVoice/en/cv-corpus-6.1-2020-12-11/en/val_dataset.csv', batch_size=64)
wer, cer = asr.evaluate.calculate_error_rates(pipeline, test_dataset)
print(f'WER: {wer}   CER: {cer}')