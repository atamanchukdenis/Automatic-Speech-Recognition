import os
import numpy as np
import tensorflow as tf
import automatic_speech_recognition as asr

#running only on GTX 1080 ("1") until RTX 2060 ("0") is fixed
os.environ["CUDA_VISIBLE_DEVICES"]="1"

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
#config = tf.config.experimental.set_memory_growth(physical_devices[1], True)

dataset = asr.dataset.Audio.from_csv(
    '/home/datamanc/data/CommonVoice/en/cv-corpus-6.1-2020-12-11/en/train_dataset.csv',
    batch_size=32,
    is_sorted=False,
    is_bins_behaviour=False,
    is_homogeneous=False,
    is_pad_transcript=True,
    bins=0)
dev_dataset = asr.dataset.Audio.from_csv('/home/datamanc/data/CommonVoice/en/cv-corpus-6.1-2020-12-11/en/val_dataset.csv',
    batch_size=32,
    is_sorted=False,
    is_bins_behaviour=False,
    is_homogeneous=False,
    is_pad_transcript=False,
    bins=0)
alphabet = asr.text.Alphabet(lang='en')
#lm = asr.text.LanguageModel('/home/datamanc/data/CommonCrawl/400K_3-gram.binary').load()
#features_extractor = asr.features.FilterBanks(
#    features_num=80,
#    winlen=0.02,
#    winstep=0.01,
#    winfunc=np.hanning
#)
features_extractor = asr.features.LogMelLibrosa(
    features_num=80,
    winlen=0.02,
    winstep=0.01,
    winfunc=np.hanning
)
model = asr.model.get_deepspeech2(
    input_dim=80,
    output_dim=28,
    is_mixed_precision=False
)
optimizer = tf.optimizers.Adam(
    lr=1e-4,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8
)
decoder = asr.decoder.GreedyDecoder()
#decoder = asr.decoder.prefix_beam_search_decoder
spec_augment = asr.augmentation.SpecAugment(
    F=40,
    mf=1,
    Tmin=10,
    Tmax=30,
    mt=5
)
pipeline = asr.pipeline.CTCPipeline(
    alphabet, features_extractor, model, optimizer, decoder,
    checkpoint_dir=None
)
pipeline.fit(dataset, dev_dataset, epochs=1, augmentation=None)
pipeline.save('checkpoint')

test_dataset = asr.dataset.Audio.from_csv('/home/datamanc/data/CommonVoice/en/cv-corpus-6.1-2020-12-11/en/val_dataset.csv',
    batch_size=32,
    is_sorted=False,
    is_bins_behaviour=False,
    is_homogeneous=False,
    is_pad_transcript=False,
    bins=0)
wer, cer = asr.evaluate.calculate_error_rates(pipeline, test_dataset)
print(f'Val WER: {wer}   Val CER: {cer}')

test_dataset = asr.dataset.Audio.from_csv('/home/datamanc/data/CommonVoice/en/cv-corpus-6.1-2020-12-11/en/train_subset_dataset.csv',
    batch_size=32,
    is_sorted=False,
    is_bins_behaviour=False,
    is_homogeneous=False,
    is_pad_transcript=False,
    bins=0)
wer, cer = asr.evaluate.calculate_error_rates(pipeline, test_dataset)
print(f'Train WER: {wer}   Train CER: {cer}')
