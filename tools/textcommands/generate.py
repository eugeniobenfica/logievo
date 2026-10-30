#!/usr/bin/env python3
"""Training sentences for a Logievo text-commands model, at scale.

Reads a command list (commands.txt: "names | more names = what to do") and writes a file the
Dataset node's "Text file" source reads, and `logievo_run --train --data` too:

    @command calculadora | calculator = calc.exe      (class 1, 2... in order)
    1<TAB>abre a calculadora pra mim
    0<TAB>não abra a calculadora

Class 0 is "do nothing". About 45% of the sentences ask for one of the commands, in Portuguese
and English, with many verbs, politeness, fillers and framings. The rest are what the model must
refuse: the same names negated, other verbs on them (close, delete...), questions and statements
about them, things that already happened or might, programs that are not in the list, things
that are not programs at all, dangerous requests, small talk and fragments. On top, noise like
Whisper's: accents dropped, homophones ("abri" for "abre"), fillers, repeated words, typos.

--split test keeps every fifth phrasing, filler, other verb and unknown thing out of training and
uses only those, so a test file measures how the model does on wordings it never saw. The verbs
that ask to open are the model's vocabulary and stay in both: add yours to OPEN_VERBS.

Only the standard library: python generate.py --count 200000 --out train.tsv
"""

from __future__ import annotations

import argparse
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# --- what asks for a command ------------------------------------------------------------------

OPEN_VERBS = [
    "abra", "abre", "abrir", "inicia", "inicie", "iniciar", "executa", "execute", "executar", "roda", "rode", "rodar",
    "carrega", "carregue", "liga", "ligue", "mostra", "mostre", "me mostra", "me abre", "acessa", "acesse", "entra no",
    "vai pro", "bota", "coloca", "puxa", "chama", "dispara", "sobe",
]
OPEN_VERBS_EN = ["open", "launch", "start", "run", "show me", "bring up", "fire up", "pull up", "load", "go to", "get me"]
ASKERS = [
    "", "", "", "", "por favor", "ei", "ok", "computador", "você pode", "pode", "consegue", "dá pra", "quero que você",
    "preciso que você", "será que você pode", "faz um favor e", "bora", "agora", "rápido", "assistente", "me ajuda e",
]
ASKERS_EN = ["", "", "", "please", "hey", "ok computer", "can you", "could you", "would you", "i want you to", "go ahead and", "quickly"]
FRAMES = ["{}", "{}", "{}", "o {}", "a {}", "o programa {}", "o app {}", "o aplicativo {}", "meu {}", "minha {}", "o tal do {}", "aquele {}"]
FRAMES_EN = ["{}", "{}", "the {}", "my {}", "the {} app", "the {} program"]
TAILS = ["", "", "", "", "pra mim", "por favor", "agora", "rapidinho", "aí", "por gentileza", "agora mesmo", "logo", "se puder", "obrigado"]
TAILS_EN = ["", "", "", "please", "now", "for me", "right now", "thanks"]
WANTS = [
    "quero usar {}", "quero abrir {}", "preciso de {}", "preciso abrir {}", "preciso usar {}", "bora abrir {}", "vamos abrir {}",
    "quero ver {}", "me dá {}", "eu queria abrir {}", "gostaria de abrir {}", "tem como abrir {}", "preciso entrar em {}",
]
WANTS_EN = ["i need {}", "i want to use {}", "i'd like to open {}", "let's open {}", "time to open {}", "i need to get into {}"]
# A name alone ("calculadora") is not a request: the model needs a verb that asks to open.

# --- what must do nothing ---------------------------------------------------------------------

NEGATORS = [
    "não", "nao", "nunca", "jamais", "não precisa", "não quero que você", "não é pra", "nem pense em", "pare de",
    "não vai", "melhor não", "não deixa", "esquece, não",
]
NEGATORS_EN = ["don't", "do not", "never", "no need to", "please don't", "stop trying to", "you don't need to"]
# Any verb but opening: so the model learns that a name with a verb it does not know does nothing.
OTHER_ACTIONS = [
    "feche", "fecha", "fechar", "apague", "apaga", "deletar", "delete", "exclua", "excluir", "desinstale", "desinstalar",
    "atualiza", "atualize", "minimiza", "minimize", "renomeia", "move", "copia", "imprime", "esconde", "trava", "mata",
    "encerra", "encerre", "desliga", "reinstala", "configura", "fecha e apaga", "quebra", "conserta", "vende", "compra",
    "lava", "pinta", "esquece", "lembra de", "odeio", "xinga", "joga fora", "baixa de novo", "limpa", "formata", "zera",
    "reseta", "trancar", "bloqueia", "derruba", "corrompe", "estraga", "come", "esvazia", "manda embora", "remove",
    "tira", "arranca", "destrói", "elimina", "sai do", "para o", "pausa", "testa", "analisa", "compara", "critica",
]
OTHER_ACTIONS_EN = [
    "close", "delete", "uninstall", "update", "minimize", "rename", "move", "copy", "hide", "kill", "quit", "remove",
    "break", "fix", "sell", "buy", "wipe", "reset", "block", "crash", "destroy", "erase", "stop", "pause", "forget", "clean",
]
ABOUT = [
    "o que é {}", "o que é o {}", "como funciona {}", "para que serve {}", "onde fica {}", "quem criou {}", "{} é bom",
    "eu gosto de {}", "eu odeio {}", "{} travou", "{} está lento", "por que {} não abre", "como eu abro {}",
    "você sabe abrir {}", "como abrir {}", "{} é pago", "qual a versão {}", "{} tem vírus", "tô cansado de {}",
    "meu chefe abriu {}", "ele sempre abre {}", "a {} quebrou", "falei de {} ontem", "lembra de {}",
]
ABOUT_EN = ["what is {}", "how does {} work", "how do i open {}", "is {} installed", "i hate {}", "{} crashed", "who made {}", "where is {}"]
HAPPENED = [
    "eu abri {} ontem", "eu já abri {}", "você abriu {}", "o joão abriu {}", "ela abriu {}", "abriram {}", "tinha aberto {}",
    "se eu abrir {}", "quando eu abrir {}", "talvez eu abra {} depois", "amanhã eu abro {}", "antes eu abria {}", "abri {} sem querer",
]
HAPPENED_EN = ["i opened {} yesterday", "i already opened {}", "she opened {}", "if i open {}", "maybe later i'll open {}"]
NOT_PROGRAMS = [
    "porta", "janela", "portão", "boca", "olhos", "conta no banco", "geladeira", "livro", "chamado", "garrafa", "guarda chuva",
    "cortina", "pacote", "presente", "caixa", "loja", "mercado", "torneira", "empresa", "cerveja", "coração", "mente",
    "carta", "envelope", "gaveta", "armário", "porta malas", "vinho", "lata", "jogo do brasil", "caminho", "parêntese",
    "the door", "the window", "a bottle", "my eyes", "a bank account", "the box", "the fridge", "a can", "the gate",
]
OTHER_APPS = [
    "photoshop", "spotify", "word", "excel", "powerpoint", "outlook", "teams", "zoom", "discord", "steam", "whatsapp",
    "telegram", "netflix", "youtube", "vs code", "visual studio", "blender", "obs", "vlc", "skype", "minecraft",
    "fortnite", "instagram", "tiktok", "chrome", "firefox", "edge", "slack", "notion", "figma", "audacity", "gimp",
]
DANGEROUS = [
    "desliga o computador", "reinicia o computador", "formata o disco", "apaga tudo", "deleta a pasta windows",
    "apaga meus documentos", "desinstala tudo", "manda um email pro chefe", "compra isso", "transfere dinheiro",
    "shut down the computer", "delete everything", "format the drive", "restart now", "rm rf", "del system32",
]
SMALL_TALK = [
    "bom dia", "boa tarde", "boa noite", "obrigado", "valeu", "que horas são", "qual a previsão do tempo", "conta uma piada",
    "tudo bem", "toca uma música", "liga a luz", "apaga a luz", "qual é o seu nome", "você é inteligente", "estou com fome",
    "nada não", "deixa pra lá", "cancela", "esquece", "espera", "um momento", "deixa eu pensar", "testando", "alô", "oi",
    "hello", "thanks", "what time is it", "never mind", "cancel", "wait", "good morning", "how are you", "testing one two",
]
FRAGMENTS = ["", "abra", "abre", "abrir", "abre aí", "abra o", "abre a", "o", "a", "hum", "é", "hã", "tipo", "open", "open the", "start", "por favor"]
# A name with no verb: not a request (Whisper heard only part of it, or it was said in passing).
BARE = ["{}", "{}", "{} {}", "{} né", "o {}", "a {}", "{} hein", "{} {} {}", "the {}", "{}?"]
SYLLABLES = ["ba", "ce", "di", "fo", "gu", "la", "me", "ni", "po", "ru", "sa", "te", "vi", "xo", "za", "tra", "pre", "quin", "lho", "nha"]
FILLERS = ["é", "hã", "hum", "tipo", "então", "olha", "bom", "ah", "ei", "uh", "um", "so", "like"]

COMMON = {"", "{}", "o {}", "a {}", "meu {}", "minha {}", "the {}", "my {}", "por favor", "pra mim", "pode", "please", "now", "agora"}

# Whisper hears these alike.
SOUND_ALIKE = {
    "abre": ["abri", "abra"], "abra": ["abre", "abri"], "pra": ["para", "pro"], "para": ["pra"], "você": ["voce", "cê", "vc"],
    "por favor": ["porfavor", "por fav", "pfv"], "programa": ["programma"], "calculadora": ["calculador", "calcula dora"],
    "notas": ["nota"], "explorador": ["exploradô"], "navegador": ["navegado"], "configurações": ["configuração"],
    "não": ["num", "nao"], "agora": ["agr"], "aplicativo": ["app", "aplicativa"],
}
KEYBOARD = "qwertyuiopasdfghjklzxcvbnm"


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def norm(s: str) -> str:
    return " ".join("".join(c if c.isalnum() else " " for c in strip_accents(s).lower()).split())


def read_commands(path: Path) -> tuple[list[str], list[list[str]]]:
    """The command lines as written, and each one's names."""
    lines, names = [], []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        left, target = line.split("=", 1)
        found = [n.strip() for n in left.split("|") if norm(n)]
        if found and target.strip():
            lines.append(line)
            names.append(found)
    if not lines:
        sys.exit(f"{path}: no commands (lines like: calculadora | calculator = calc.exe)")
    return lines, names


class Generator:
    def __init__(self, names: list[list[str]], split: str, seed: int, noise: float):
        self.rng = random.Random(seed)
        self.names = names
        self.noise = noise
        self.test = split == "test"
        taken = {norm(n) for ns in names for n in ns}

        def part(items: list[str]) -> list[str]:
            # Every fifth item belongs to the test split only, except the everyday glue every
            # request has ("a", "pra mim", "por favor"): unseen, it would test nothing real.
            if len(items) < 5:
                return items
            held = [i % 5 == 4 and x not in COMMON for i, x in enumerate(items)]
            kept = [x for x, h in zip(items, held) if h == self.test]
            return kept or items

        self.open_verbs, self.open_verbs_en = OPEN_VERBS, OPEN_VERBS_EN
        self.askers, self.askers_en = part(ASKERS), part(ASKERS_EN)
        self.frames, self.frames_en = part(FRAMES), part(FRAMES_EN)
        self.tails, self.tails_en = part(TAILS), part(TAILS_EN)
        self.wants, self.wants_en = part(WANTS), part(WANTS_EN)
        self.negators, self.negators_en = part(NEGATORS), part(NEGATORS_EN)
        self.other_actions, self.other_actions_en = part(OTHER_ACTIONS), part(OTHER_ACTIONS_EN)
        self.about, self.about_en = part(ABOUT), part(ABOUT_EN)
        self.happened, self.happened_en = part(HAPPENED), part(HAPPENED_EN)
        # Unknown things and apps: never one of the commands' names.
        self.not_programs = [x for x in part(NOT_PROGRAMS) if norm(x) not in taken]
        self.other_apps = [x for x in part(OTHER_APPS) if norm(x) not in taken]
        self.dangerous, self.small_talk = part(DANGEROUS), part(SMALL_TALK)

    def pick(self, items):
        return items[self.rng.randrange(len(items))]

    def english(self) -> bool:
        return self.rng.random() < 0.25

    def thing(self, name: str, en: bool) -> str:
        return self.pick(self.frames_en if en else self.frames).format(name)

    def join(self, *parts: str) -> str:
        return " ".join(p for p in parts if p)

    def request(self, name: str) -> str:
        """Asking for `name`, in one of many ways."""
        en = self.english()
        if self.rng.random() < 0.25:
            return self.pick(self.wants_en if en else self.wants).format(self.thing(name, en))
        if en:
            return self.join(self.pick(self.askers_en), self.pick(self.open_verbs_en), self.thing(name, True), self.pick(self.tails_en))
        return self.join(self.pick(self.askers), self.pick(self.open_verbs), self.thing(name, False), self.pick(self.tails))

    def refusal(self) -> str:
        """Something that must do nothing."""
        r = self.rng.random()
        en = self.english()
        name = self.pick(self.pick(self.names))
        if r < 0.22:  # the same request, negated
            if en:
                return self.join(self.pick(self.negators_en), self.pick(self.open_verbs_en), self.thing(name, True))
            return self.join(self.pick(self.negators), self.pick(self.open_verbs), self.thing(name, False), self.pick(self.tails))
        if r < 0.40:  # another verb on it
            if en:
                return self.join(self.pick(self.askers_en), self.pick(self.other_actions_en), self.thing(name, True))
            return self.join(self.pick(self.askers), self.pick(self.other_actions), self.thing(name, False), self.pick(self.tails))
        if r < 0.55:  # a question or a statement about it
            return self.pick(self.about_en if en else self.about).format(self.thing(name, en))
        if r < 0.63:  # it happened, or might
            return self.pick(self.happened_en if en else self.happened).format(self.thing(name, en))
        if r < 0.75 and self.other_apps:  # a program that is not in the list: asked properly, still nothing
            return self.request(self.pick(self.other_apps))
        if r < 0.85 and self.not_programs:  # opening what is not a program
            return self.request(self.pick(self.not_programs))
        if r < 0.89:
            return self.pick(self.dangerous)
        if r < 0.94:
            return self.pick(self.small_talk)
        if r < 0.98:
            return self.pick(BARE).replace("{}", name, 1).replace("{}", self.pick(self.tails))
        return self.pick(FRAGMENTS)

    def made_up(self) -> str:
        return "".join(self.pick(SYLLABLES) for _ in range(self.rng.randrange(1, 4)))

    def noisy(self, s: str, name: str = "") -> str:
        """How Whisper (or a hurried typist) might give it. The words of `name` are never dropped:
        a request whose program went unheard would teach the wrong answer."""
        rng = self.rng
        if not self.test and rng.random() < 0.3:
            # Words it never saw, anywhere: the answer must not hang on the words it knows to be
            # unknown ("será que", "assistente", a name Whisper mangled...).
            words = s.split()
            for _ in range(rng.randrange(1, 3)):
                words.insert(rng.randrange(len(words) + 1), self.made_up())
            s = " ".join(words)
        if rng.random() > self.noise:
            return s
        keep = set(name.split())
        words = s.split()
        for _ in range(1 + (rng.random() < 0.3)):
            k = rng.random()
            if k < 0.2:
                s = strip_accents(" ".join(words))
                words = s.split()
            elif k < 0.4:
                text = " ".join(words)
                for a, bs in SOUND_ALIKE.items():
                    if f" {a} " in f" {text} " and rng.random() < 0.6:
                        text = f" {text} ".replace(f" {a} ", f" {rng.choice(bs)} ", 1).strip()
                words = text.split()
            elif k < 0.55 and words:
                words.insert(rng.randrange(len(words) + 1), rng.choice(FILLERS))
            elif k < 0.65 and words:
                i = rng.randrange(len(words))
                words.insert(i, words[i])  # a repeated word
            elif k < 0.75 and len(words) > 2:
                droppable = [i for i, w in enumerate(words) if w not in keep]
                if droppable:
                    words.pop(rng.choice(droppable))  # a word not heard
            else:
                long = [i for i, w in enumerate(words) if len(w) >= 4]
                if long:
                    i = rng.choice(long)
                    w = words[i]
                    j = rng.randrange(1, len(w) - 1)
                    op = rng.randrange(4)
                    if op == 0:
                        w = w[:j] + w[j + 1:]
                    elif op == 1:
                        w = w[:j] + w[j] + w[j:]
                    elif op == 2:
                        w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
                    else:
                        w = w[:j] + rng.choice(KEYBOARD) + w[j + 1:]
                    words[i] = w
        s = " ".join(words)
        r = rng.random()
        if r < 0.2:
            s = s.capitalize()
        elif r < 0.25:
            s = s.upper()
        if rng.random() < 0.3:
            s += rng.choice([".", "?", "!", "...", ","])
        return s

    def examples(self, count: int, open_share: float):
        per = len(self.names)
        for i in range(count):
            if self.rng.random() < open_share:
                k = i % per  # every command as often
                name = self.pick(self.names[k])
                yield k + 1, self.noisy(self.request(name), name)
            else:
                yield 0, self.noisy(self.refusal())


def write(path: Path, lines: list[str], names: list[list[str]], count: int, split: str, seed: int, noise: float, open_share: float, unique: bool) -> Counter:
    gen = Generator(names, split, seed, noise)
    rows = list(gen.examples(count, open_share))
    if unique:
        seen, kept = set(), []
        for label, s in rows:
            key = norm(s)
            if key not in seen:
                seen.add(key)
                kept.append((label, s))
        rows = kept
    # The same words with two answers would only teach doubt: the refusal wins.
    answer: dict[str, int] = {}
    for label, s in rows:
        key = norm(s)
        if answer.setdefault(key, label) != label:
            answer[key] = 0
    rows = [(answer[norm(s)], s) for _, s in rows]
    gen.rng.shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"# Logievo text commands: {len(rows)} examples, split {split}, seed {seed}, noise {noise}\n")
        f.write("# made by tools/textcommands/generate.py; class 0 does nothing, class k is the k-th @command\n")
        for line in lines:
            f.write(f"@command {line}\n")
        for label, s in rows:
            f.write(f"{label}\t{s.replace(chr(9), ' ')}\n")
    return Counter(label for label, _ in rows)


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--commands", type=Path, default=here / "commands.txt", help="names = what to do, one command per line")
    p.add_argument("--out", type=Path, required=True, help="the examples file (.tsv) to write")
    p.add_argument("--count", type=int, default=200_000)
    p.add_argument("--split", choices=["train", "test"], default="train", help="test: only phrasings that training never sees")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--noise", type=float, default=0.35, help="share of sentences given Whisper-like noise")
    p.add_argument("--open-share", type=float, default=0.45, help="share of sentences that ask for a command")
    p.add_argument("--unique", action="store_true", help="drop repeated sentences")
    a = p.parse_args(argv)
    lines, names = read_commands(a.commands)
    counts = write(a.out, lines, names, a.count, a.split, a.seed, a.noise, a.open_share, a.unique)
    total = sum(counts.values())
    print(f"{a.out}: {total} examples ({a.split}), {len(lines)} commands", file=sys.stderr)
    print(f"  do nothing: {counts[0]} ({100 * counts[0] / max(1, total):.1f}%)", file=sys.stderr)
    for k, ns in enumerate(names, 1):
        print(f"  {k}. {ns[0]}: {counts[k]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
