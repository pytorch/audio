# Speech Recognition with wav2vec2.0

This example demonstarates how you can use torchaudio's I/O features and models to run speech recognition in C++ application.

## 1. Create a transcription pipeline TorchScript file

We will create a TorchScript that performs the following processes;

1. Load audio from a file.
1. Pass audio to encoder which produces the sequence of probability distribution on labels.
1. Pass the encoder output to decoder which generates transcripts.

For building decoder, we borrow the pre-trained weights published by `fairseq` and/or Hugging Face Transformers, then convert it `torchaudio`'s format, which supports TorchScript.

### 1.1. From `fairseq`

For `fairseq` models, you can download pre-trained weights
You can download a model from [`fairseq` repository](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec). Here, we will use `Base / 960h` model. You also need to download [the letter dictionary file](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec#evaluating-a-ctc-model).

For the decoder part, we use [simple_ctc](https://github.com/mthrok/ctcdecode), which also supports TorchScript.

```bash
mkdir -p pipeline-fairseq
python build_pipeline_from_fairseq.py \
    --model-file "wav2vec_small_960.pt" \
    --dict-dir <DIRECTORY_WHERE_dict.ltr.txt_IS_FOUND> \
    --output-path "./pipeline-fairseq/"
```

The above command should create the following TorchScript object files in the output directory.

```
decoder.zip  encoder.zip  loader.zip
```

* `loader.zip` loads audio file and generate waveform Tensor.
* `encoder.zip` receives waveform Tensor and generates the sequence of probability distribution over the label.
* `decoder.zip` receives the probability distribution over the label and generates a transcript.

### 1.2. From Hugging Face Transformers


[Hugging Face Transformers](https://huggingface.co/transformers/index.html) and [Hugging Face Model Hub](https://huggingface.co/models) provides `wav2vec2.0` models fine-tuned on variety of datasets and languages.

We can also import the model published on Hugging Face Hub and run it in our C++ application.
In the following example, we will try the Geremeny model, ([facebook/wav2vec2-large-xlsr-53-german](https://huggingface.co/facebook/wav2vec2-large-xlsr-53-german/tree/main)) on [VoxForge Germany dataset](http://www.voxforge.org/de/downloads).

```bash
mkdir -p pipeline-hf
python build_pipeline_from_huggingface_transformers.py \
    --model facebook/wav2vec2-large-xlsr-53-german \
    --output-path ./pipeline-hf/
```

The resulting TorchScript object files should be same as the `fairseq` example.

## 2. Build the application

Please refer to [the top level README.md](../README.md)

## 3. Run the application

Now we run the C++ application [`transcribe`](./transcribe.cpp), with the TorchScript object we created in Step.1.1. and an input audio file.

```bash
../build/speech_recognition/transcribe ./pipeline-fairseq ../data/input.wav
```

This will output something like the following.

```
Loading module from: ./pipeline/loader.zip
Loading module from: ./pipeline/encoder.zip
Loading module from: ./pipeline/decoder.zip
Loading the audio
Running inference
Generating the transcription
I HAD THAT CURIOSITY BESIDE ME AT THIS MOMENT
Done.
```

## 4. Evaluate the pipeline on Librispeech dataset

Let's evaluate this word error rate (WER) of this application using [Librispeech dataset](https://www.openslr.org/12).

### 4.1. Create a list of audio paths

For the sake of simplifying our C++ code, we will first parse the Librispeech dataset to get the list of audio path

```bash
python parse_librispeech.py <PATH_TO_YOUR_DATASET>/LibriSpeech/test-clean ./flist.txt
```

The list should look like the following;

```bash
head flist.txt

1089-134691-0000    /LibriSpeech/test-clean/1089/134691/1089-134691-0000.flac    HE COULD WAIT NO LONGER
```

### 4.2. Run the transcription

[`transcribe_list`](./transcribe_list.cpp) processes the input flist list and feed the audio path one by one to the pipeline, then generate reference file and hypothesis file.

```bash
../build/speech_recognition/transcribe_list ./pipeline-fairseq ./flist.txt <OUTPUT_DIR>
```

### 4.3. Score WER

You need `sclite` for this step. You can download the code from [SCTK repository](https://github.com/usnistgov/SCTK).

```bash
# in the output directory
sclite -r ref.trn -h hyp.trn -i wsj -o pralign -o sum
```

WER can be found in the resulting `hyp.trn.sys`. Check out the column that starts with `Sum/Avg` the first column of the third block is `100 - WER`.

In our test, we got the following results.

|          model                            | Fine Tune | test-clean | test-other |
|:-----------------------------------------:|----------:|:----------:|:----------:|
| Base<br/>`wav2vec_small_960`              |      960h |        3.1 |        7.7 |
| Large<br/>`wav2vec_big_960`               |      960h |        2.6 |        5.9 |
| Large (LV-60)<br/>`wav2vec2_vox_960h_new` |      960h |        2.9 |        6.2 |
| Large (LV-60) + Self Training<br/>`wav2vec_vox_960h_pl` | 960h | 1.9 |      4.5 |


You can also check `hyp.trn.pra` file to see what errors were made.

```
id: (3528-168669-0005)
Scores: (#C #S #D #I) 7 1 0 0
REF:  there is a stone to be RAISED heavy
HYP:  there is a stone to be RACED  heavy
Eval:                        S
```

## 5. Evaluate the pipeline on VoxForge dataset

Now we use the pipeline we created in step 1.2. This time with German language dataset from VoxForge.

### 5.1. Create a list of audio paths

Download an archive from http://www.repository.voxforge1.org/downloads/de/Trunk/Audio/Main/16kHz_16bit/, and extract it to your local file system, then run the following to generate the file list.

```bash
python parse_voxforge.py <PATH_TO_YOUR_DATASET> > ./flist-de.txt
```

The list should look like

```bash
head flist-de.txt
de5-001    /datasets/voxforge/de/guenter-20140214-afn/wav/de5-001.wav    ES SOLL ETWA FÜNFZIGTAUSEND VERSCHIEDENE SORTEN GEBEN
```

### 5.2. Run the application and score WER

This process is same as the Librispeech example. We just use the pipeline with the Germany model and file list of Germany dataset. Refer to the corresponding ssection in Librispeech evaluation..

```bash
../build/speech_recognition/transcribe_list ./pipeline-hf ./flist-de.txt <OUTPUT_DIR>
```

Then

```bash
# in the output directory
sclite -r ref.trn -h hyp.trn -i wsj -o pralign -o sum
```

You can find the detail of evalauation result in PRA.

```
id: (guenter-20140214-afn/mfc/de5-012)
Scores: (#C #S #D #I) 4 1 1 0
REF:  die ausgaben kÖnnen gigantisch STEIGE N
HYP:  die ausgaben kÖnnen gigantisch ****** STEIGEN
Eval:                                 D      S
```
