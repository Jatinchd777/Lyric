# lyric

Generate timed `.lrc` lyrics from any audio file using [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

```
lyric <artist> <title> <audio>
```

## How it works

1. Decodes the audio to 16 kHz mono via `ffmpeg`
2. Transcribes it with Whisper (`large-v3-turbo`, GPU + flash attention)
3. Writes `[mm:ss.xx]` timestamped lines to `~/.local/share/rmpc/lyrics/<artist> - <title>.lrc`

The output lands directly in [rmpc](https://github.com/mierak/rmpc)'s lyrics directory.

## Requirements

- CMake ≥ 3.20 and a C++17 compiler
- `ffmpeg` on `PATH`
- A built copy of `whisper.cpp` in `whisper.cpp/` (Vulkan backend recommended)
- Whisper model at `~/.local/share/lyric/models/ggml-large-v3-turbo.bin`

## Build

```sh
git clone --recursive https://github.com/ggml-org/whisper.cpp
cmake -B whisper.cpp/build -S whisper.cpp -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build whisper.cpp/build -j

cmake -B build
cmake --build build -j
```

## Usage

```sh
./build/lyric "Radiohead" "No Surprises" music.mp3
# ~/.local/share/rmpc/lyrics/Radiohead - No Surprises.lrc
```

The generated path is printed to stdout; progress goes to stderr.

## Tuning

These are hardcoded in `src/main.cpp`:

| What | Where |
| --- | --- |
| Model path | `MODEL_PATH` |
| Output directory | `LYRICS_DIR` |
| Timing offset | `OFFSET` |
| GPU device | `context_params.gpu_device` |
