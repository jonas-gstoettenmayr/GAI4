# Homework  5 - Jonas Gstöttenmayr

I choose the cold war as my topic, it is one of the more interesting topics, has a bit of onscreen text and is not in 4k like some other clips. [CLIP](https://commons.wikimedia.org/wiki/File:Der_Kalte_Krieg_-_Planet_Wissen.webm)

## Homework models

### Summary

While accurate, the cold war is quite a well documented era as such it can be called into question how much the LLM truly summarized the text it recieved and how much it simply generated from its own "knowledge". But for the summary itself it is accurate. The only true complaint is the length, as the video is already quite short and thus the transcript, having a summary of such length does not really feel like a summary of the topic (but at least it does not miss anything important ¯\_(ツ)_/¯ ) - For reference the summary has ~500 characters the original text has ~1500.

#### The *summary*

```Text
The Cold War, beginning after World War II in 1945, was a prolonged conflict between the United States and the Soviet Union characterized by threats and arms buildup rather than direct combat. This rivalry, spanning 45 years, involved ideological competition between communism and capitalism, as well as a race for global influence through military and technological advancements, including nuclear weapons and space exploration. The Cold War ended with the reunification of Germany and the dissolution of the Soviet Union in 1991. However, it resumed in 2022 with Russia's invasion of Ukraine.
```

### Translation

For the translation with the original code a small bug became appearent, the LLM was passed the text snippet by snipet (split by time), but the splits where in the middle of sentences, which the LLM does not know. As such the translation quality has suffered quite a bit as the LLM builds two distinct sentences out of one long sentence that was accidentially split.

Example:

German: Es beginnt ein aggressiver Wettstreit |SPLIT| der Systeme, der 45 Jahre andauert.
English: An aggressive competition begins. |SPLIT| the systems that lasted for 45 years.

The LLm is not at fault here as it could not know what comes after, but it does lower the overall quality of the translation. A solution could be to simply merge the text before translation (actualy happens) and remove the newlines (does not).

<br>

### OCR

The OCR is really intersting for my clip as it not only captures the onscreen text but also the text of the signs in the background. For example at 18 seconds It is able to deciver the protestors signs with superb accuracy even for text where I myself struggle to read what is actually written on there. For some reason 3 seconds later it classifes the text slightly differently even though the picture didn't change but that is only for that one second and it gets back to the original good guess the next frame again.

Altough I can't judge the accuracy in the caes of it reading the soviet propaganda poster I am still impressed by the fact tha it can read the cyrilic from the poster to a very good degree. What is interesting here is that for one frame it decides to read far more of the poster than in the other having a sentence twice as long.

At the end with the protestor signs it has a bit of trublle with the overlaid text, reading "Kill puting, save us" as "Kill us".

---

## Smaller models

To minimze variables and code changes I left everything with src the same other the the model strings in models.py (so altough the class is still named QwenChat it uses the smaller model in the backend). I also exchanged how the model loading class to AutoModelForImageTextToText.

For my models I used whisper-tiny for the transcription and SmolVLM2-2.2B-Instruct for the summary and OCR.

The whisper tiny performance is far faster then the whisper-large v3 from the EX, while keeping up the same quality. It also groups the text a bit better with the majorty beeing just in the first sentence. The only real mistake I found was in the alst sentece with it writing "Frussland"
 instead of "Russland".

The SmolVLM2 is a different story, it is far worse than the counter part. The translation failed completely, then only correct part is the first sentence. Afterwards it halucinates its own cold ware story and even degenerates into repeating the same sentence over and over again.

Furthermore in the lower parts it deviates from its systemprompt and starts rambaling on about how it would translate the sentence over just translating it.

With the catastrophic translation the sumamary is also a complete failure. I tried with the previous Exs translation (that was actually good), with a proper translation the summary also succeeded to a good degree, while not perfect it was a passable summary, though notacibly worse then the quen one.

Altough disapointed the results are not totatlly unexpected considering the size of the model and its particular focus on the visual aspect.

Durring the OCR it does slightly better but still not good, it only picks up on the perfect injected texts like "WDR", "Der Kalte Krieg" while missing any and all stylised text, or any text on posters/pictures inside the videa. But at least its 6 times faster then the qwen model...

## Specs comparision

### whisper large v3

Times:

- Transcription - 2.25 minutes

```bash
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.261.03             Driver Version: 535.261.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       On  | 00000001:00:00.0 Off |                  Off |
| N/A   70C    P0              82W /  70W |  11054MiB / 16384MiB |    100%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

```

### whisper-tiny

Time:

- Transcription - 17 seconds

```bash
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.261.03             Driver Version: 535.261.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       On  | 00000001:00:00.0 Off |                  Off |
| N/A   45C    P0              47W /  70W |    835MiB / 16384MiB |     40%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```

### GWEN2.5-VL 7B

Times:

- Translation - 1 minute
- Summary - 40 seconds
- OCR - 12.5 minutes

For the OCR (higher V-RAM then e.g. summary as video is also loaded)

```bash
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.261.03             Driver Version: 535.261.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       On  | 00000001:00:00.0 Off |                  Off |
| N/A   70C    P0              82W /  70W |   8823MiB / 16384MiB |    100%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```

### SmolVLM2-2.2B-Instruct

Times:

- Translation -  1 minute
- Summary - 9 seconds
- OCR - 2 minutes

For the OCR (higher V-RAM then e.g. summary as video is also loaded)

```bash
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.261.03             Driver Version: 535.261.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       On  | 00000001:00:00.0 Off |                  Off |
| N/A   48C    P0             100W /  70W |   2025MiB / 16384MiB |     34%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```