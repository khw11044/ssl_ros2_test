



```
pip install deepfilternet
```


```
┌────────────┐
│ edie_mic   │  (raw stereo, 48k)
│  node      │
└─────┬──────┘
      │
      ├──────────────▶ SSL node
      │                (GCC / SRP)
      │
      └──▶ deepfilter_node
              |
              v
        vad_node ─▶ stt_node

```