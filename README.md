[中文](README-MAC.md) | [English](README.md) 

# Kokoro TextToSpeech Node for ComfyUI

![image](https://github.com/billwuhao/ComfyUI_KokoroTTS_MW/blob/master/images/2025-03-05_17-09-35.png)

## 📣 Update

[2025-03-22] ⚒️: Refactored code, faster generation speed. 

[2025-03-05]⚒️: Supports 8 languages and 150 voices.

- New language support added: 

'e' => Spanish  
'f' => French  
'h' => Hindi   
'i' => Italian   
'p' => Brazilian Portuguese   

- Corresponding newly added voices: 

"e": ["ef_dora.pt",
        "em_alex.pt",
        "em_santa.pt"]

"f": ["ff_siwis.pt"]

"h": ["hf_alpha.pt",
        "hf_beta.pt",
        "hm_omega.pt",
        "hm_psi.pt"]

"i": ["if_sara.pt",
        "im_nicola.pt"]

"p": ["pf_dora.pt",
"pm_alex.pt",
"pm_santa.pt"]

- Add 100 new Chinese voices:

['zf_001.pt', 'zf_002.pt', 'zf_003.pt', 'zf_004.pt', 'zf_005.pt', 'zf_006.pt', 'zf_007.pt', 'zf_008.pt', 'zf_017.pt', 'zf_018.pt', 'zf_019.pt', 'zf_021.pt', 'zf_022.pt', 'zf_023.pt', 'zf_024.pt', 'zf_026.pt', 'zf_027.pt', 'zf_028.pt', 'zf_032.pt', 'zf_036.pt', 'zf_038.pt', 'zf_039.pt', 'zf_040.pt', 'zf_042.pt', 'zf_043.pt', 'zf_044.pt', 'zf_046.pt', 'zf_047.pt', 'zf_048.pt', 'zf_049.pt', 'zf_051.pt', 'zf_059.pt', 'zf_060.pt', 'zf_067.pt', 'zf_070.pt', 'zf_071.pt', 'zf_072.pt', 'zf_073.pt', 'zf_074.pt', 'zf_075.pt', 'zf_076.pt', 'zf_077.pt', 'zf_078.pt', 'zf_079.pt', 'zf_083.pt', 'zf_084.pt', 'zf_085.pt', 'zf_086.pt', 'zf_087.pt', 'zf_088.pt', 'zf_090.pt', 'zf_092.pt', 'zf_093.pt', 'zf_094.pt', 'zf_099.pt', 'zm_009.pt', 'zm_010.pt', 'zm_011.pt', 'zm_012.pt', 'zm_013.pt', 'zm_014.pt', 'zm_015.pt', 'zm_016.pt', 'zm_020.pt', 'zm_025.pt', 'zm_029.pt', 'zm_030.pt', 'zm_031.pt', 'zm_033.pt', 'zm_034.pt', 'zm_035.pt', 'zm_037.pt', 'zm_041.pt', 'zm_045.pt', 'zm_050.pt', 'zm_052.pt', 'zm_053.pt', 'zm_054.pt', 'zm_055.pt', 'zm_056.pt', 'zm_057.pt', 'zm_058.pt', 'zm_061.pt', 'zm_062.pt', 'zm_063.pt', 'zm_064.pt', 'zm_065.pt', 'zm_066.pt', 'zm_068.pt', 'zm_069.pt', 'zm_080.pt', 'zm_081.pt', 'zm_082.pt', 'zm_089.pt', 'zm_091.pt', 'zm_095.pt', 'zm_096.pt', 'zm_097.pt', 'zm_098.pt', 'zm_100.pt']


## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_KokoroTTS_MW.git
cd ComfyUI_KokoroTTS_MW
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## Models Download

- The models and voices need to be manually downloaded and placed in the `ComfyUI\models\Kokorotts` path.

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)  
[Kokoro-82M-v1.1-zh](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh)

The structure is as follows:
```
ComfyUI\models\Kokorotts
│ Kokoro-82M
   └── voices
   config.json
   kokoro-v1_0.pth
| Kokoro-82M-v1.1-zh
   └── voices
   config.json
   kokoro-v1_1-zh.pth
```

---

![image](https://github.com/billwuhao/ComfyUI_KokoroTTS_MW/blob/master/images/2025-02-17_01-39-16.png)

## Features

- High-quality text-to-speech synthesis
- Multiple voice options
- Support for multilingual text
- Easy integration with ComfyUI workflows

## Supported Language

'a' => American English 

'b' => British English 

'j' => Japanese 

'z' => Chinese 

## Available Voices

"a": 

["af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", 
   "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", 
   "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", 
   "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa"]

"b": 

["bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel",
   "bm_fable", "bm_george", "bm_lewis"]

"j": 

["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"]

"z": 

["zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
   "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"]

### Acknowledgement

- [Kokoro](https://github.com/hexgrad/kokoro)