import math
import struct
import wave

sample_rate = 16000
seconds = 2
amp = 0.4

with wave.open('sample_audio.wav', 'wb') as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(sample_rate)
    frames = []

    for i in range(sample_rate * seconds):
        t = i / sample_rate
        sample_value = int(32767 * amp * math.sin(2 * math.pi * 440 * t))
        frames.append(struct.pack('<h', sample_value))

    wav.writeframes(b''.join(frames))

print('Created sample_audio.wav')
