from pipersynth import PiperPipeline

with PiperPipeline.from_pretrained("en_US-lessac-medium") as pipe:
    pipe("Hello from the first call.").save_wav("one.wav")
    pipe("The same model session is reused.").save_wav("two.wav")
