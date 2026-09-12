from pipersynth import PiperVoice, SynthesisConfig

voice = PiperVoice.load("voice.onnx")
voice.save_wav(
    "out.wav",
    "Hello from PiperSynth.",
    SynthesisConfig(length_scale=1.0),
)
