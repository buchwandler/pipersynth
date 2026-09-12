from pipersynth import synthesize_to_wav

output = synthesize_to_wav(
    "Hello, this sentence was generated with PiperSynth.",
    "hello.wav",
    voice="en_US-lessac-medium",
)

print(f"Wrote {output}")
