from pipersynth import PipelineConfig, PiperPipeline

with PiperPipeline(PipelineConfig(model_path="voice.onnx")) as pipe:
    result = pipe.run("Hello world.")
    result.save_wav("hello.wav")
