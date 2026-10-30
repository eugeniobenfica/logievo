// logievo_run: a trained text-commands model on the command line. An exported model is this
// program with the model appended (core/Export.h).
//
//   model.exe [--dry-run] [--threshold 0.6] [--model project.logievo] [sentence...]
//
// With no sentence it reads one line from the input (pipe Whisper's text into it). It does what
// the sentence asks for, or nothing, and prints nothing: a command's target is opened (a program,
// a file, a folder or a link), or with "run: program arguments" that command line is run.
// --dry-run prints what it would do ("-" for nothing) and its confidence, and does nothing.
// Exit code: 0 done, 1 nothing to do, 2 an error (said on the error output).
//
// With no window at all, the whole pipeline (tools/textcommands/pipeline.py drives it):
//
//   logievo_run --train project.logievo [--data examples.tsv] [--epochs N] [--batch N] [--lr X] [--export model.exe]
//       trains the project (made from the Text Commands template when it does not exist yet; with
//       --data its dataset reads that file, and its last layer gets a score for every command),
//       saves it with what it learned, and exports it.
//   model.exe --eval examples.tsv
//       how many of the file's examples the model gets right, and the ones it gets wrong.

#include "core/Export.h"
#include "core/Serialization.h"
#include "core/Templates.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace logievo;

namespace {

std::filesystem::path fromUtf8(const std::string& s) { return std::filesystem::path(std::u8string(s.begin(), s.end())); }

#ifdef _WIN32
std::string utf8(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
    return s;
}
std::wstring wide(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

// The arguments in UTF-8 (argv is in the local code page, which loses accents).
std::vector<std::string> arguments(int, char**) {
    int count = 0;
    wchar_t** list = CommandLineToArgvW(GetCommandLineW(), &count);
    std::vector<std::string> args;
    for (int i = 1; i < count; ++i) args.push_back(utf8(list[i]));
    LocalFree(list);
    return args;
}

std::filesystem::path self(const char*) {
    std::wstring buf(32768, L'\0');
    buf.resize(GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size())));
    return std::filesystem::path(buf);
}

std::wstring expand(const std::wstring& t) {
    std::wstring out(32768, L'\0');
    const DWORD n = ExpandEnvironmentStringsW(t.c_str(), out.data(), static_cast<DWORD>(out.size()));
    return n > 0 && n <= out.size() ? out.substr(0, n - 1) : t;
}

bool shellOpen(const std::wstring& file, const std::wstring& params) {
    return reinterpret_cast<INT_PTR>(ShellExecuteW(nullptr, L"open", file.c_str(), params.empty() ? nullptr : params.c_str(), nullptr, SW_SHOWNORMAL)) > 32;
}

// A command line, in a console of its own (a window program shows none), not waited for.
bool run(const std::string& command) {
    std::wstring line = expand(wide(command));
    STARTUPINFOW si{};
    si.cb = sizeof si;
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, line.data(), nullptr, nullptr, FALSE, CREATE_NEW_CONSOLE, nullptr, nullptr, &si, &pi)) return false;
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return true;
}

// A program, file, folder or link, with %VARIABLES% expanded; "program arguments" also works.
bool open(const std::string& target) {
    const std::wstring t = expand(wide(target));
    if (shellOpen(t, {})) return true;
    const size_t space = t.find(L' ');
    return space != std::wstring::npos && shellOpen(t.substr(0, space), t.substr(space + 1));
}
#else
std::vector<std::string> arguments(int argc, char** argv) { return {argv + 1, argv + argc}; }

std::filesystem::path self(const char* argv0) {
    std::error_code ec;
    const auto p = std::filesystem::read_symlink("/proc/self/exe", ec);
    return ec ? std::filesystem::path(argv0) : p;
}

bool spawn(const char* file, const char* a, const char* b, const char* c, bool wait) {
    const pid_t pid = fork();
    if (pid == 0) {
        execlp(file, a, b, c, static_cast<char*>(nullptr));
        _exit(127);
    }
    if (pid < 0) return false;
    int status = 0;
    return !wait || (waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

bool run(const std::string& command) { return spawn("/bin/sh", "sh", "-c", command.c_str(), false); }

bool open(const std::string& target) {
#ifdef __APPLE__
    return spawn("open", "open", target.c_str(), nullptr, true);
#else
    return spawn("xdg-open", "xdg-open", target.c_str(), nullptr, true);
#endif
}
#endif

// "run: program arguments" runs it; anything else is opened.
bool perform(const std::string& target) {
    if (target.rfind("run:", 0) == 0) {
        const size_t start = target.find_first_not_of(' ', 4);
        return start != std::string::npos && run(target.substr(start));
    }
    return open(target);
}

int error(const std::string& why) {
    std::fprintf(stderr, "%s\n", why.c_str());
    return 2;
}

// --train: the whole training with no window, progress on the error output.
int train(const std::filesystem::path& project, const std::string& data, const TrainSettings& set, const std::filesystem::path& exportTo,
          const std::filesystem::path& runner) {
    Graph g;
    Weights w;
    std::string why;
    std::error_code ec;
    if (std::filesystem::exists(project, ec)) {
        if (!loadGraph(project, g, &why, &w)) return error(why);
    } else {
        g = makeTemplate("commands", project.stem().string());
    }
    if (!data.empty()) {
        int sources = 0;
        for (Node& n : g.nodes)
            if (n.type == "dataset") {
                ++sources;
                n.params["source"] = kTextFileSource;
                const auto file = std::filesystem::absolute(fromUtf8(data), ec).u8string();
                n.params["file"] = std::string(file.begin(), file.end());
            }
        if (sources != 1) return error("The project should have one Dataset to read the file");
        if (!fitOutputToClasses(g)) return error("The project's Loss should score a Dense layer against the dataset's targets");
    }
    // What the command line gives (0: the project's own).
    if (set.epochs > 0) g.training.epochs = set.epochs;
    if (set.batchSize > 0) g.training.batchSize = set.batchSize;
    if (set.learningRate > 0) g.training.learningRate = set.learningRate;
    for (const Node& n : g.nodes)
        if (n.type == "dataset") {
            const auto d = loadDataset(g, n);
            if (!d->error.empty()) return error(d->error);
            std::fprintf(stderr, "%lld examples, %lld classes, %d epochs\n", static_cast<long long>(d->rows), static_cast<long long>(d->classCount),
                         g.training.epochs);
        }
    Trainer trainer;
    if (!trainer.start(g, w)) return error(trainer.report().error);
    int shown = 0;
    const auto began = std::chrono::steady_clock::now();
    while (trainer.running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const TrainReport r = trainer.report();
        for (; shown < r.epoch; ++shown) {
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
            std::fprintf(stderr, "epoch %d/%d  loss %.4f  accuracy %.2f%%  (%.0fs)\n", shown + 1, r.epochs, r.loss[static_cast<size_t>(shown)],
                         shown < static_cast<int>(r.accuracy.size()) ? r.accuracy[static_cast<size_t>(shown)] * 100 : 0.0, seconds);
        }
    }
    const TrainReport r = trainer.report();
    if (!r.error.empty()) return error(r.error);
    trainer.takeWeights(w);
    if (!saveGraph(project, g, &why, &w)) return error(why);
    std::fprintf(stderr, "saved %s\n", project.string().c_str());
    if (!exportTo.empty()) {
        if (!exportExecutable(g, w, runner, exportTo, &why)) return error(why);
        std::fprintf(stderr, "exported %s\n", exportTo.string().c_str());
    }
    return 0;
}

// --eval: right answers on a file of examples at the threshold, how other thresholds trade acting
// wrongly for missing, and some of the wrong ones.
int evaluate(const CommandModel& model, const std::filesystem::path& file, float threshold) {
    std::ifstream in(file, std::ios::binary);
    if (!in) return error("Could not open " + file.string());
    struct Example {
        std::string sentence;
        int expected;  // -1: do nothing
        Decision best; // its pick with no threshold
    };
    std::vector<Example> examples;
    std::string line, list;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        if (line.rfind("@command ", 0) == 0) {
            list += line.substr(9) + "\n";
            continue;
        }
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        const std::string sentence = line.substr(tab + 1);
        examples.push_back({sentence, std::atoi(line.c_str()) - 1, decide(model, sentence, 0.0f)});
    }
    if (examples.empty()) return error("No examples in " + file.string());
    if (parseCommands(list).size() != model.commands.size())
        std::printf("note: the file lists %zu commands, the model %zu\n", parseCommands(list).size(), model.commands.size());
    auto pick = [](const Example& e, float t) { return e.best.confidence >= t ? e.best.command : -1; };
    std::printf("%zu examples\n threshold   right    acted wrongly   did nothing when asked   wrong command\n", examples.size());
    for (const float t : {0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 0.95f}) {
        int64_t right = 0, falseRuns = 0, missed = 0, wrong = 0, refusals = 0, asks = 0;
        for (const Example& e : examples) {
            const int got = pick(e, t);
            (e.expected < 0 ? refusals : asks) += 1;
            if (got == e.expected) ++right;
            else if (e.expected < 0) ++falseRuns; // the dangerous kind: it acts when it should not
            else if (got < 0) ++missed;
            else ++wrong;
        }
        auto pct = [](int64_t a, int64_t b) { return b ? 100.0 * static_cast<double>(a) / static_cast<double>(b) : 0.0; };
        std::printf(" %s %.2f    %6.2f%%  %6.2f%% (%lld)     %6.2f%% (%lld)            %lld\n", t == threshold ? ">" : " ", t,
                    pct(right, static_cast<int64_t>(examples.size())), pct(falseRuns, refusals), static_cast<long long>(falseRuns), pct(missed, asks),
                    static_cast<long long>(missed), static_cast<long long>(wrong));
    }
    std::printf("(acted wrongly: of the sentences that should do nothing; did nothing: of the requests)\n");
    int shown = 0;
    for (const Example& e : examples) {
        const int got = pick(e, threshold);
        if (got == e.expected || shown++ >= 12) continue;
        auto name = [&](int c) { return c < 0 ? std::string("-") : model.commands[static_cast<size_t>(c)].target; };
        std::printf("  \"%s\": %s (%.2f), expected %s\n", e.sentence.c_str(), name(got).c_str(), e.best.confidence, name(e.expected).c_str());
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
#endif
    bool dryRun = false;
    float threshold = 0.6f;
    TrainSettings set;
    set.epochs = set.batchSize = 0;
    set.learningRate = 0;
    std::string modelFile, sentence, trainProject, data, exportTo, evalFile;
    const std::vector<std::string> args = arguments(argc, argv);
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        const bool more = i + 1 < args.size();
        if (a == "--dry-run") dryRun = true;
        else if (a == "--threshold" && more) threshold = std::strtof(args[++i].c_str(), nullptr);
        else if (a == "--model" && more) modelFile = args[++i];
        else if (a == "--train" && more) trainProject = args[++i];
        else if (a == "--data" && more) data = args[++i];
        else if (a == "--epochs" && more) set.epochs = std::atoi(args[++i].c_str());
        else if (a == "--batch" && more) set.batchSize = std::atoi(args[++i].c_str());
        else if (a == "--lr" && more) set.learningRate = std::strtod(args[++i].c_str(), nullptr);
        else if (a == "--export" && more) exportTo = args[++i];
        else if (a == "--eval" && more) evalFile = args[++i];
        else if (a == "--help" || a == "-h") {
            std::printf("usage: %s [--dry-run] [--threshold 0.6] [--model project.logievo] [sentence...]\n"
                        "       %s --eval examples.tsv\n"
                        "       logievo_run --train project.logievo [--data examples.tsv] [--epochs N] [--batch N] [--lr X] [--export model.exe]\n",
                        self(argv[0]).filename().string().c_str(), self(argv[0]).filename().string().c_str());
            return 0;
        } else sentence += (sentence.empty() ? "" : " ") + a;
    }
    if (!trainProject.empty()) return train(fromUtf8(trainProject), data, set, exportTo.empty() ? std::filesystem::path() : fromUtf8(exportTo), self(argv[0]));

    Graph graph;
    Weights weights;
    std::string why;
    const bool loaded = modelFile.empty() ? readEmbeddedModel(self(argv[0]), graph, weights, &why) : loadGraph(fromUtf8(modelFile), graph, &why, &weights);
    if (!loaded) return error(why);
    CommandModel model;
    if (!prepareCommandModel(graph, weights, model, &why)) return error(why);
    if (!evalFile.empty()) return evaluate(model, fromUtf8(evalFile), threshold);

    if (sentence.empty()) std::getline(std::cin, sentence);
    const Decision d = decide(model, sentence, threshold);
    if (dryRun) {
        std::printf("%s\t%.3f\n", d.command < 0 ? "-" : model.commands[static_cast<size_t>(d.command)].target.c_str(), d.confidence);
        return d.command < 0 ? 1 : 0;
    }
    if (d.command < 0) return 1;
    const std::string& target = model.commands[static_cast<size_t>(d.command)].target;
    return perform(target) ? 0 : error("Could not do " + target);
}
