#include "core/Dataset.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <random>

namespace logievo {

namespace {

// Saved by number in projects: new sets go at the end.
enum Source { Circles, Spirals, Xor, Sine, Csv, Line, Sequence, Commands, TextFile };
static_assert(Commands == kTextCommandsSource && TextFile == kTextFileSource);
constexpr int64_t kSequenceSteps = 16;
constexpr double kPi = 3.14159265358979323846;

std::filesystem::path fromUtf8(const std::string& s) { return std::filesystem::path(std::u8string(s.begin(), s.end())); }

void synthetic(Dataset& d, int source, int64_t samples, float noise) {
    std::mt19937 rng(7); // the same set every time
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    d.rows = samples;
    if (source == Sequence) {
        // Stretches of waves of different speeds and phases; the target is the value that
        // comes next.
        d.features = kSequenceSteps;
        d.classes = false;
        for (int64_t i = 0; i < samples; ++i) {
            const float speed = 0.2f + (uni(rng) + 1.0f) * 0.25f, phase = uni(rng) * static_cast<float>(kPi);
            for (int64_t k = 0; k < kSequenceSteps; ++k) d.x.push_back(std::sin(phase + speed * static_cast<float>(k)) + gauss(rng) * noise * 0.1f);
            d.y.push_back(std::sin(phase + speed * static_cast<float>(kSequenceSteps)));
        }
        return;
    }
    d.features = source == Sine || source == Line ? 1 : 2;
    d.classes = source != Sine && source != Line;
    d.classCount = source == Spirals ? 3 : d.classes ? 2 : 0;
    for (int64_t i = 0; i < samples; ++i) {
        float a = 0, b = 0, label = 0;
        if (source == Circles) {
            // Two rings, one inside the other.
            const int cls = static_cast<int>(i % 2);
            const float angle = (uni(rng) + 1.0f) * static_cast<float>(kPi);
            const float radius = (cls ? 0.45f : 1.0f) + gauss(rng) * noise * 0.5f;
            a = radius * std::cos(angle);
            b = radius * std::sin(angle);
            label = static_cast<float>(cls);
        } else if (source == Spirals) {
            // Three arms winding out from the middle.
            const int cls = static_cast<int>(i % 3);
            const float t = static_cast<float>(i / 3) / static_cast<float>(std::max<int64_t>(1, samples / 3));
            const float angle = cls * 2.0f * static_cast<float>(kPi) / 3.0f + t * 4.0f + gauss(rng) * noise;
            a = t * std::cos(angle);
            b = t * std::sin(angle);
            label = static_cast<float>(cls);
        } else if (source == Xor) {
            a = uni(rng) + gauss(rng) * noise * 0.3f;
            b = uni(rng) + gauss(rng) * noise * 0.3f;
            label = (a > 0) != (b > 0) ? 1.0f : 0.0f;
        } else if (source == Line) {
            // y = 2x - 1, give or take: what a linear regression finds.
            a = uni(rng) * 2.0f;
            label = 2.0f * a - 1.0f + gauss(rng) * noise;
        } else {
            a = uni(rng) * static_cast<float>(kPi);
            label = std::sin(a) + gauss(rng) * noise * 0.3f;
        }
        d.x.push_back(a);
        if (d.features == 2) d.x.push_back(b);
        d.y.push_back(label);
    }
}

void readCsv(Dataset& d, const std::filesystem::path& file, bool targetFirst, bool classes) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        d.error = file.empty() ? "Choose a CSV file" : "Could not open " + file.filename().string();
        return;
    }
    std::string line;
    int64_t lineNo = 0;
    std::vector<float> row;
    while (std::getline(in, line)) {
        ++lineNo;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.find_first_not_of(" \t") == std::string::npos) continue;
        row.clear();
        bool numeric = true;
        size_t start = 0;
        while (start <= line.size()) {
            size_t end = line.find_first_of(",;\t", start);
            if (end == std::string::npos) end = line.size();
            const std::string cell = line.substr(start, end - start);
            char* stop = nullptr;
            const float v = std::strtof(cell.c_str(), &stop);
            if (stop == cell.c_str()) numeric = false;
            row.push_back(v);
            start = end + 1;
        }
        if (!numeric) {
            if (d.rows == 0 && lineNo == 1) continue; // a header row
            d.error = "Line " + std::to_string(lineNo) + " has something that is not a number";
            return;
        }
        if (row.size() < 2) {
            d.error = "Needs at least two columns: features and a target";
            return;
        }
        const int64_t features = static_cast<int64_t>(row.size()) - 1;
        if (d.rows == 0) d.features = features;
        else if (features != d.features) {
            d.error = "Line " + std::to_string(lineNo) + " has " + std::to_string(row.size()) + " columns, the first had " + std::to_string(d.features + 1);
            return;
        }
        const float target = targetFirst ? row.front() : row.back();
        d.x.insert(d.x.end(), row.begin() + (targetFirst ? 1 : 0), row.end() - (targetFirst ? 0 : 1));
        d.y.push_back(target);
        ++d.rows;
    }
    if (d.rows == 0) {
        d.error = "The file has no rows";
        return;
    }
    d.classes = classes;
    if (classes) {
        for (float t : d.y) {
            if (t < 0 || t != std::floor(t)) {
                d.error = "Targets should be class numbers 0, 1, 2... (or choose Numbers)";
                return;
            }
            d.classCount = std::max<int64_t>(d.classCount, static_cast<int64_t>(t) + 1);
        }
    }
}

// --- text commands: made-up requests, in Portuguese and English. About half ask to open one of
// the commands (its class); the rest must do nothing (class 0): the same names negated, other
// verbs on them, asking to open things that are not programs, and small talk.

const std::vector<std::string> kOpen{
    "abra {}", "abra o {}", "abra a {}", "abrir {}", "abrir o {}", "abre {}", "abre o {}", "abre a {}", "abre ai o {}",
    "pode abrir {}", "pode abrir o {} pra mim", "por favor abra {}", "abra {} por favor", "quero abrir {}", "quero abrir o {}",
    "eu quero usar o {}", "preciso do {}", "preciso abrir a {}", "inicia {}", "iniciar o {}", "inicie a {}", "executa {}",
    "execute o {}", "roda o {}", "carrega o {}", "me abre {}", "me abre a {}", "liga o {}", "mostra o {}", "abrir {} agora",
    "voce pode abrir o {}", "{}", "open {}", "open the {}", "launch {}", "start {}", "run {}", "please open {}", "can you open {}",
    "open up the {}"};
const std::vector<std::string> kNegated{
    "nao abra {}", "nao abra o {}", "nao abre a {}", "nao quero abrir {}", "nao quero o {}", "nunca abra {}", "nao precisa abrir o {}",
    "nao abre o {} nao", "nao execute {}", "nao inicie o {}", "melhor nao abrir a {}", "don't open {}", "do not open the {}",
    "never open {}", "don't start {}"};
const std::vector<std::string> kOtherVerbs{
    "feche {}", "fecha o {}", "fechar a {}", "apague {}", "apaga o {}", "deletar {}", "exclua o {}", "desinstale {}",
    "desinstalar o {}", "minimiza o {}", "o que e {}", "o que e o {}", "como funciona o {}", "para que serve a {}",
    "eu nao gosto do {}", "o {} travou", "a {} esta lenta", "eu abri o {} ontem", "voce abriu a {}", "como abrir o {}",
    "quem fez o {}", "atualiza o {}", "close {}", "close the {}", "delete {}", "uninstall {}", "what is {}", "how does {} work",
    "i opened {} yesterday"};
const std::vector<std::string> kNotPrograms{
    "porta", "janela", "portao", "boca", "conta no banco", "geladeira", "livro", "chamado", "garrafa", "guarda chuva", "cortina",
    "pacote", "olho", "presente", "caixa", "loja", "mercado", "torneira", "empresa", "cerveja", "the door", "the window",
    "a bottle", "my eyes", "a bank account", "the box"};
const std::vector<std::string> kSmallTalk{
    "bom dia", "boa noite", "obrigado", "valeu", "que horas sao", "qual a previsao do tempo", "conta uma piada", "tudo bem",
    "toca uma musica", "liga a luz", "desliga o computador", "reinicia o computador", "apaga tudo", "formata o disco",
    "abra", "abre ai", "abrir", "hmm", "nada", "deixa pra la", "cancela", "esquece", "hello", "thanks", "what time is it",
    "open", "never mind", "shut down the computer", ""};
const std::vector<std::string> kFillers{"por favor", "ai", "agora", "hum", "ei", "rapidinho", "please", "now", "hey"};

std::string fill(const std::string& phrase, const std::string& name) {
    const size_t at = phrase.find("{}");
    return at == std::string::npos ? phrase : phrase.substr(0, at) + name + phrase.substr(at + 2);
}

// A slip in one longer word: a letter dropped, doubled or swapped with the next.
std::string typo(std::string s, std::mt19937& rng) {
    auto wordEnd = [&](size_t i) { return std::min(s.find(' ', i), s.size()); };
    std::vector<size_t> starts;
    for (size_t i = 0; i < s.size(); ++i)
        if ((i == 0 || s[i - 1] == ' ') && wordEnd(i) - i >= 4) starts.push_back(i);
    if (starts.empty()) return s;
    const size_t w = starts[rng() % starts.size()];
    const size_t len = wordEnd(w) - w;
    const size_t at = w + 1 + rng() % (len - 2);
    switch (rng() % 3) {
    case 0: s.erase(at, 1); break;
    case 1: s.insert(at, 1, s[at]); break;
    default: std::swap(s[at], s[at + 1]);
    }
    return s;
}

void textCommands(Dataset& d, const std::string& list, int64_t samples, float noise) {
    const std::vector<TextCommand> commands = parseCommands(list);
    if (commands.empty()) {
        d.error = "Write at least one command: names = what to open";
        return;
    }
    // Things that are not programs, unless the list makes them one.
    std::vector<std::string> others;
    for (const std::string& o : kNotPrograms) {
        bool named = false;
        for (const TextCommand& c : commands)
            for (const std::string& n : c.names) named |= normalizeText(n) == normalizeText(o);
        if (!named) others.push_back(o);
    }
    std::mt19937 rng(7); // the same set every time
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    auto pick = [&](const std::vector<std::string>& v) -> const std::string& { return v[rng() % v.size()]; };
    d.rows = samples;
    d.features = kTextFeatures;
    d.classes = true;
    d.classCount = 1 + static_cast<int64_t>(commands.size());
    d.commands = list;
    d.texts.reserve(static_cast<size_t>(samples));
    for (int64_t i = 0; i < samples; ++i) {
        const size_t k = rng() % commands.size();
        const std::string& name = pick(commands[k].names);
        const double r = uni(rng);
        std::string s;
        float label = 0;
        if (r < 0.5) {
            s = fill(pick(kOpen), name);
            label = static_cast<float>(k + 1);
        } else if (r < 0.65) s = fill(pick(kNegated), name);
        else if (r < 0.8) s = fill(pick(kOtherVerbs), name);
        else if (r < 0.9 && !others.empty()) s = fill(pick(kOpen), pick(others));
        else s = pick(kSmallTalk);
        if (uni(rng) < 0.25) s = uni(rng) < 0.5 ? pick(kFillers) + " " + s : s + " " + pick(kFillers);
        s = normalizeText(s);
        if (uni(rng) < noise) s = typo(s, rng);
        d.texts.push_back(std::move(s));
        d.y.push_back(label);
    }
}

void readTextFile(Dataset& d, const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        d.error = file.empty() ? "Choose a text file (tools/textcommands/generate.py makes one)" : "Could not open " + file.filename().string();
        return;
    }
    std::string line, list;
    int64_t lineNo = 0, top = 0;
    while (std::getline(in, line)) {
        ++lineNo;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.find_first_not_of(" \t") == std::string::npos || line[0] == '#') continue;
        if (line.rfind("@command ", 0) == 0) {
            list += line.substr(9) + "\n";
            continue;
        }
        const size_t tab = line.find('\t');
        char* stop = nullptr;
        const long label = std::strtol(line.c_str(), &stop, 10);
        if (tab == std::string::npos || stop != line.c_str() + tab || label < 0) {
            d.error = "Line " + std::to_string(lineNo) + " should be a class number, a tab and the sentence";
            return;
        }
        d.texts.push_back(normalizeText(line.substr(tab + 1)));
        d.y.push_back(static_cast<float>(label));
        top = std::max<int64_t>(top, label);
    }
    const size_t commands = parseCommands(list).size();
    if (commands == 0) d.error = "The file lists no commands: \"@command names = what to open\" lines";
    else if (d.texts.empty()) d.error = "The file has no examples";
    else if (top > static_cast<int64_t>(commands))
        d.error = "An example has class " + std::to_string(top) + ", the file lists " + std::to_string(commands) + " commands";
    if (!d.error.empty()) {
        d.texts.clear();
        d.y.clear();
        return;
    }
    d.rows = static_cast<int64_t>(d.texts.size());
    d.features = kTextFeatures;
    d.classes = true;
    d.classCount = 1 + static_cast<int64_t>(commands);
    d.commands = list;
}

uint32_t fnv1a(const std::string& s) {
    uint32_t h = 2166136261u;
    for (unsigned char c : s) h = (h ^ c) * 16777619u;
    return h;
}

std::string trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

void scale(Dataset& d, int how) {
    if (how == 2)
        for (float& v : d.x) v /= 255.0f;
    else if (how == 1)
        for (int64_t f = 0; f < d.features; ++f) {
            double sum = 0, sq = 0;
            for (int64_t r = 0; r < d.rows; ++r) sum += d.x[static_cast<size_t>(r * d.features + f)];
            const double mean = sum / static_cast<double>(d.rows);
            for (int64_t r = 0; r < d.rows; ++r) sq += std::pow(d.x[static_cast<size_t>(r * d.features + f)] - mean, 2);
            const double sd = std::sqrt(sq / static_cast<double>(d.rows));
            for (int64_t r = 0; r < d.rows; ++r) {
                float& v = d.x[static_cast<size_t>(r * d.features + f)];
                v = static_cast<float>((v - mean) / (sd > 1e-8 ? sd : 1.0));
            }
        }
}

} // namespace

std::shared_ptr<const Dataset> loadDataset(const Graph& g, const Node& n) {
    const int source = static_cast<int>(g.number(n, "source"));
    const std::string file = g.text(n, "file");
    std::string key = std::to_string(source) + "|" + std::to_string(static_cast<int>(g.number(n, "scale")));
    if (source == Csv || source == TextFile) {
        std::error_code ec;
        const auto path = fromUtf8(file);
        const auto stamp = std::filesystem::last_write_time(path, ec);
        key += "|" + file + "|" + std::to_string(ec ? 0 : stamp.time_since_epoch().count()) + "|" + std::to_string(static_cast<int>(g.number(n, "column"))) +
               "|" + std::to_string(static_cast<int>(g.number(n, "targets")));
    } else {
        key += "|" + std::to_string(static_cast<int64_t>(g.number(n, "samples"))) + "|" + std::to_string(g.number(n, "noise"));
        if (source == Commands) key += "|" + g.text(n, "commands");
    }
    static std::mutex mutex;
    static std::map<std::string, std::shared_ptr<const Dataset>> cache;
    {
        std::lock_guard lock(mutex);
        if (auto it = cache.find(key); it != cache.end()) return it->second;
    }
    auto d = std::make_shared<Dataset>();
    if (source == Csv) readCsv(*d, fromUtf8(file), g.number(n, "column") == 1.0, g.number(n, "targets") == 0.0);
    else if (source == TextFile) readTextFile(*d, fromUtf8(file));
    else if (source == Commands) textCommands(*d, g.text(n, "commands"), static_cast<int64_t>(g.number(n, "samples")), static_cast<float>(g.number(n, "noise")));
    else synthetic(*d, source, static_cast<int64_t>(g.number(n, "samples")), static_cast<float>(g.number(n, "noise")));
    // Text features stay as they are: an exported model reads sentences with no dataset at hand.
    if (d->error.empty() && d->texts.empty()) scale(*d, static_cast<int>(g.number(n, "scale")));
    std::lock_guard lock(mutex);
    if (cache.size() > 16) cache.clear();
    cache[key] = d;
    return d;
}

const char* const kDefaultCommands =
    "# One command per line: names (separated by |) = what to open: a program, a file, a folder or a link\n"
    "bloco de notas | notepad | notas = notepad.exe\n"
    "calculadora | calculator = calc.exe\n"
    "paint = mspaint.exe\n"
    "explorador de arquivos | gerenciador de arquivos | file explorer = explorer.exe\n"
    "documentos | meus documentos | pasta de documentos = %USERPROFILE%\\Documents\n"
    "terminal | prompt de comando | cmd = cmd.exe\n"
    "navegador | internet | browser = https://www.google.com\n";

std::vector<TextCommand> parseCommands(const std::string& text) {
    std::vector<TextCommand> list;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find('\n', start);
        if (end == std::string::npos) end = text.size();
        const std::string line = trim(text.substr(start, end - start));
        start = end + 1;
        const size_t eq = line.find('=');
        if (line.empty() || line[0] == '#' || eq == std::string::npos) continue;
        TextCommand c;
        c.target = trim(line.substr(eq + 1));
        const std::string names = line.substr(0, eq);
        size_t a = 0;
        while (a <= names.size()) {
            size_t b = names.find('|', a);
            if (b == std::string::npos) b = names.size();
            if (std::string n = trim(names.substr(a, b - a)); !normalizeText(n).empty()) c.names.push_back(n);
            a = b + 1;
        }
        if (!c.names.empty() && !c.target.empty()) list.push_back(std::move(c));
    }
    return list;
}

std::string normalizeText(const std::string& text) {
    // U+00C0..U+00FF (two bytes, the first 0xC3) folded to plain letters.
    static const char latin1[] = "aaaaaaaceeeeiiiidnooooo ouuuuy saaaaaaaceeeeiiiidnooooo ouuuuy y";
    std::string out;
    for (size_t i = 0; i < text.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        char k = ' ';
        if (c < 0x80) {
            if (std::isalnum(c)) k = static_cast<char>(std::tolower(c));
        } else if (c == 0xC3 && i + 1 < text.size() && (static_cast<unsigned char>(text[i + 1]) & 0xC0) == 0x80) {
            k = latin1[static_cast<unsigned char>(text[++i]) - 0x80];
        } else {
            while (i + 1 < text.size() && (static_cast<unsigned char>(text[i + 1]) & 0xC0) == 0x80) ++i; // another letter: a space
        }
        if (k != ' ' || (!out.empty() && out.back() != ' ')) out += k;
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

std::vector<float> textFeatures(const std::string& text) {
    std::vector<float> f(static_cast<size_t>(kTextFeatures), 0.0f);
    auto put = [&](const std::string& key, float v) {
        float& x = f[fnv1a(key) % static_cast<uint32_t>(kTextFeatures)];
        x = std::min(1.0f, x + v);
    };
    const std::string t = normalizeText(text);
    std::string prev;
    size_t start = 0;
    while (start < t.size()) {
        size_t end = t.find(' ', start);
        if (end == std::string::npos) end = t.size();
        const std::string w = t.substr(start, end - start);
        start = end + 1;
        put("w " + w, 1.0f);
        if (!prev.empty()) put("p " + prev + " " + w, 1.0f);
        // Letter triples, so a misheard or misspelled word still lands near the right one.
        const std::string padded = "<" + w + ">";
        for (size_t j = 0; j + 3 <= padded.size(); ++j) put("c " + padded.substr(j, 3), 0.34f);
        prev = w;
    }
    return f;
}

} // namespace logievo
