from pipersynth import PipelineConfig, PiperPipeline

with PiperPipeline(PipelineConfig(model_path="voice.onnx")) as pipe:
    for unit in pipe.iter_units("First sentence. Second sentence."):
        print(len(unit.audio_int16_bytes), "PCM bytes")
