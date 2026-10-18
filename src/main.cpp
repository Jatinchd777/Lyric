#include "whisper.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

constexpr double OFFSET = 0.37;

const fs::path LYRICS_DIR =
    fs::path(std::getenv("HOME")) / ".local/share/rmpc/lyrics";

const fs::path MODEL_PATH = fs::path(std::getenv("HOME")) /
                            ".local/share/lyric/models/ggml-large-v3-turbo.bin";

std::string sanitize(const std::string &input) {
    std::string value = input;

    if (const auto pos = value.find(','); pos != std::string::npos) {
        value.erase(pos);
    }

    constexpr std::string_view illegal = R"(/\":*?<>|{})";

    for (char &c : value) {
        if (illegal.find(c) != std::string_view::npos) {
            c = '_';
        }
    }

    value.erase(std::remove(value.begin(), value.end(), '\''), value.end());

    const auto first = value.find_first_not_of(" \t\n\r\f\v");
    const auto last = value.find_last_not_of(" \t\n\r\f\v");

    if (first == std::string::npos) {
        return "Unknown";
    }

    value = value.substr(first, last - first + 1);

    return value.empty() ? "Unknown" : value;
}

std::string lrc_timestamp(double seconds) {
    const int minutes = static_cast<int>(seconds / 60.0);

    seconds -= minutes * 60.0;

    char buffer[32];

    std::snprintf(buffer, sizeof(buffer), "[%02d:%05.2f]", minutes, seconds);

    return buffer;
}

std::string shell_quote(const std::string &value) {
    std::string result = "'";

    for (const char c : value) {
        if (c == '\'') {
            result += "'\\''";
        } else {
            result += c;
        }
    }

    result += "'";

    return result;
}

std::vector<float> load_audio(const fs::path &audio) {
    const std::string command = "ffmpeg -v error -i " +
                                shell_quote(audio.string()) +
                                " -f f32le -ac 1 -ar 16000 pipe:1";

    FILE *pipe = popen(command.c_str(), "r");

    if (!pipe) {
        throw std::runtime_error("failed to start ffmpeg");
    }

    std::vector<float> samples;

    constexpr size_t BUFFER_SIZE = 8192;
    float buffer[BUFFER_SIZE];

    while (true) {
        const size_t count =
            std::fread(buffer, sizeof(float), BUFFER_SIZE, pipe);

        if (count > 0) {
            samples.insert(samples.end(), buffer, buffer + count);
        }

        if (count < BUFFER_SIZE) {
            break;
        }
    }

    const int status = pclose(pipe);

    if (status != 0) {
        throw std::runtime_error("ffmpeg failed to decode audio");
    }

    return samples;
}

std::string trim(const char *text) {
    if (!text) {
        return {};
    }

    std::string value(text);

    const auto first = value.find_first_not_of(" \t\n\r\f\v");

    if (first == std::string::npos) {
        return {};
    }

    const auto last = value.find_last_not_of(" \t\n\r\f\v");

    return value.substr(first, last - first + 1);
}

int main(int argc, char *argv[]) {

    if (argc != 4) {
        std::cerr << "usage: lyric <artist> <title> <audio>\n";

        return 1;
    }

    const std::string artist = argv[1];
    const std::string title = argv[2];
    const fs::path audio = argv[3];

    if (!fs::is_regular_file(audio)) {
        std::cerr << "Error: Audio file not found: " << audio << '\n';

        return 1;
    }

    const std::string safe_artist = sanitize(artist);
    const std::string safe_title = sanitize(title);

    std::error_code ec;

    fs::create_directories(LYRICS_DIR, ec);

    if (ec) {
        std::cerr << "Error: could not create lyrics directory: "
                  << ec.message() << '\n';

        return 1;
    }

    const fs::path output_path =
        LYRICS_DIR / (safe_artist + " - " + safe_title + ".lrc");

    std::cerr << "Transcribing: " << artist << " - " << title << '\n';

    try {

        const std::vector<float> samples = load_audio(audio);

        if (samples.empty()) {
            std::cerr << "Audio produced no samples.\n";

            return 1;
        }

        whisper_context_params context_params =
            whisper_context_default_params();

        context_params.use_gpu = true;

        context_params.gpu_device = 1;

        context_params.flash_attn = true;

        whisper_context *context = whisper_init_from_file_with_params(
            MODEL_PATH.c_str(), context_params);

        if (!context) {
            std::cerr << "Error: failed to initialize Whisper model.\n";

            return 1;
        }

        whisper_full_params params =
            whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);

        params.print_progress = true;
        params.print_realtime = false;
        params.print_timestamps = false;

        params.token_timestamps = false;

        params.language = "auto";
        params.detect_language = false;

        params.vad = false;

        params.beam_search.beam_size = 5;

        params.n_threads =
            static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));

        params.translate = false;

        const int result = whisper_full(context, params, samples.data(),
                                        static_cast<int>(samples.size()));

        if (result != 0) {
            std::cerr << "❌ Error: Whisper transcription failed.\n";

            whisper_free(context);

            return 1;
        }

        std::vector<std::string> lyric_lines;

        lyric_lines.emplace_back("[ar:" + artist + "]");

        lyric_lines.emplace_back("[ti:" + title + "]");

        lyric_lines.emplace_back("");

        bool found_lyrics = false;

        const int segment_count = whisper_full_n_segments(context);

        for (int i = 0; i < segment_count; ++i) {

            const char *raw_text = whisper_full_get_segment_text(context, i);

            const std::string text = trim(raw_text);

            if (text.empty()) {
                continue;
            }

            found_lyrics = true;

            const double segment_start =
                static_cast<double>(whisper_full_get_segment_t0(context, i)) /
                100.0;

            const double adjusted_start = std::max(0.0, segment_start + OFFSET);

            lyric_lines.push_back(lrc_timestamp(adjusted_start) + text);
        }

        if (!found_lyrics) {
            std::cerr << "Whisper produced no lyrics.\n";

            whisper_free(context);

            return 1;
        }

        std::ofstream output(output_path, std::ios::binary);

        if (!output) {
            std::cerr << "Error: could not write: " << output_path << '\n';

            whisper_free(context);

            return 1;
        }

        for (const auto &line : lyric_lines) {
            output << line << '\n';
        }

        output.close();

        whisper_free(context);

        std::cout << output_path << '\n';

        return 0;

    } catch (const std::exception &e) {

        std::cerr << "Error: " << e.what() << '\n';

        return 1;
    }
}
