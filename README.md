# Logievo

Build AI models from scratch as graphs of nodes, and train them right there. C++20,
Qt 6 Widgets and [bestyle](https://github.com/eugeniobenfica/bestyle), the look it
shares with bemotion (with a white accent here). The engine is plain C++, with no
machine-learning library: tensors, shapes, gradients and training are written in
`src/core`.

- **Freedom, from zero.** A new project is a blank canvas, or one of the templates: Linear
  Regression, Network from Scratch, Classifier (MLP), Recurrent (RNN), Tensors, Custom Code
  Node and Text Commands (`src/core/Templates.cpp`). The building blocks go
  down to the math: `Parameter` (a trainable tensor of any shape), `Constant`, Add,
  Subtract, Multiply, Divide (broadcasting), Matrix Multiply, Sum, Mean, Power,
  Exp, Log, Transpose, Reshape, Concatenate. A dense layer is a Parameter, a Matrix
  Multiply and an Add, if you want to build it that way. Ready-made blocks are there
  too: Dense, Conv 2D, Recurrent (an RNN, trained through its steps), Embedding, Layer Norm,
  Dropout, pooling, and the activations.
- **Data:** a `Dataset` node gives features and targets, from built-in sets
  (Circles, Spirals, XOR, Sine wave, Line, Sequences of 16 steps), your own CSV file (one example per row,
  target in the first or last column), or **Text commands**: a list of `names = what to open` lines
  becomes made-up requests in Portuguese and English. About half ask to open one of them. The rest
  must do nothing (class 0): the same names negated ("não abra"), other verbs ("feche", "o que é"),
  things that are not programs ("abra a porta") and small talk. Sentences become 4096 hashed
  word, word-pair and letter-triple features, a batch at a time. **Text file** reads the same at
  scale from a file (`@command` lines, then `class<TAB>sentence`), as made by
  `tools/textcommands` (see below).
- **Export:** a trained Text commands or Text file model becomes a command-line program. The `Export`
  button (or File › Export) copies `logievo_run` and appends the model to it, with the file's
  command list. `model.exe "abra a calculadora"` does what the sentence asks for, or nothing, and
  prints nothing. A target is opened (a program, a file, a folder or a link, with `%VARIABLES%`
  expanded). A `run: program arguments` target runs that command line instead. With no sentence it reads
  one line from the input, so Whisper's text can be piped into it. `--dry-run` prints what it would
  open and its confidence; `--threshold 0.6` is the confidence below which it does nothing. The exit
  code is 0 when it opened something, 1 when there was nothing to do, and 2 on an error. The Text
  Commands template is a ready model to train and export.
- **Training:** a `Loss` node (cross entropy, mean squared error, binary cross
  entropy) says what to minimize. Every Loss in the graph is added up. Train runs
  on its own thread with SGD or Adam and shows the loss and accuracy per epoch. What
  is learned is saved in the project.
- **Your own nodes:** a `Code` node runs a formula you write, applied value by value to
  its inputs `a b c d`. Lines can name values, and the last line is the result. Logievo
  works out the derivative (dual numbers), so it trains like any other node. Save
  formulas in Settings › Custom Nodes and they appear in the library under Custom.
- **Performance:** in Settings › Performance you pick how many processor threads the
  engine uses (a shared pool for matrix products and convolutions). You can also turn on
  the graphics card: an OpenGL 4.3 compute shader then does the big matrix products, and
  a speed test checks it against the processor.
- **As you build**, every node shows what shape comes out (`B` is the batch) and how
  many trainable values it holds, and what does not fit is marked in red.

**Home screen** (like bemotion's): your projects, search, New Project, Open, the
projects folder, Settings and Help. **Editor:** a menu bar as bemotion's (the Logievo mark, which takes you back
to the projects; File, Edit, Add, View, Model, Help; on the right Settings, Train and Export), the
node library on the left (click a tile, or drag it in: the card itself shows where it lands), the
canvas in the middle, and the inspector on the right. The inspector shows the selected node's
settings and shapes. With nothing selected, it shows the model's summary, its training and its
problems. Settings: Ctrl+, or the gear; Help: F1 or the Help menu.

**Settings:** Theme (Dark, Midnight, Light, accent, rounding), Interface (text size,
font, grid and snapping, projects folder), Performance, and Custom Nodes. **Help:**
Shortcuts, Learn, About, Credits, License.

Canvas: drag from an output to an input to connect. Drop on a card to use its free
input, or on empty space to pick a node to add there. Drag a connected input to take
its link off. Wheel zooms, and the middle button or Space + drag pans. Drag on empty
space to box-select.

| Key | Action |
| --- | --- |
| Delete | Delete the selection |
| Ctrl+D | Duplicate |
| Ctrl+Z / Ctrl+Shift+Z | Undo / redo |
| Ctrl+A | Select all |
| F | Frame everything |
| Ctrl+S | Save |
| Ctrl+Enter | Train / stop |
| Ctrl+= / Ctrl+- / Ctrl+0 | Zoom in / out / actual size |

## Build (Windows, MSYS2 UCRT64, the same setup as bemotion)

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build && ctest --test-dir build
```

Or just run `run.bat`. `build/logievo.exe` needs `C:\msys64\ucrt64\bin` on `PATH`
(run.bat sets it).

## Layout

| Folder | What |
| --- | --- |
| `src/core` | plain C++, no Qt: `Graph`, `NodeTypes` (the node table), `Analysis` (shape inference), `Tensor` (broadcasting, matrix products), `Dataset`, `Runtime` (forward, backward, `Trainer` with SGD / Adam), `Serialization` (`.logievo` JSON, weights in base64), `Templates` (what a new project starts from) |
| `src/core` (more) | `Parallel` (the thread pool and the matrix-product kernel hook), `Expression` (the Code node language, with dual-number derivatives), `Export` (a text-commands model run outside the editor, packed into the runner) |
| `src/runner` | `logievo_run`, the command-line program Export copies (statically linked, so it runs on its own); also `--train`, `--eval` |
| `tools/textcommands` | `generate.py` (training sentences at scale), `pipeline.py` (generate, train, export, evaluate), `commands.txt` |
| `examples` | `text-commands.logievo`: the Text Commands template, trained |
| `src/ui` | `HomeScreen`, `EditorWindow`, `NodeLibrary` (left), `NodeCanvas` (middle), `Inspector` (right, with the Training section), `Document` (selection, undo, weights, training), `SettingsWindow` and `HelpWindow` (on `PagedWindow` and `WindowChrome`, from bemotion), `Logo` |
| `src/app` | `Application`, `Settings` (`settings.json` in the app data folder, or `LOGIEVO_APPDATA`), `Preferences` (applies look and performance), `Gpu` (OpenGL compute), `Platform` |
| `libs/bestyle` | the shared look (copied from bemotion; `tools/bestyle.sh` once this is a git repository) |
| `tests` | the engine (every operation's gradient checked against finite differences, a training run that must learn, CSV reading) and the canvas driven with synthetic mouse and keys |

To add a node type:

1. Add it to `src/core/NodeTypes.cpp`.
2. Add its shape rule to `Analysis.cpp`.
3. Add its forward and backward to `Runtime.cpp`, and any weights to `weightShapes`.
4. Add its icon to `src/ui/Icons.cpp`.
5. Add a gradient case to `tests/CoreTests.cpp`.

Node type and parameter ids are saved in files, so never rename them.

## Text commands at scale (`tools/textcommands`, Python, standard library only)

```bash
python tools/textcommands/pipeline.py --count 200000 --epochs 6
```

1. `generate.py` writes the training sentences. It writes a test set too, made of phrasings that
   training never sees. Both come from `commands.txt` (`names | more names = target`, one command
   per line). About 45% ask for a command, in Portuguese and English. The rest must do nothing:
   negations, other verbs, questions, past and future tenses, apps that are not in the list, things
   that are not programs, dangerous requests, small talk, and bare names. Whisper-like noise is
   added on top: dropped accents, homophones, fillers, repeated or lost words, and typos. Training
   sentences also get made-up words, so unknown words do not sway the answer.
2. `logievo_run --train build/textcommands/comandos.logievo --data train.tsv --export comandos.exe`
   trains with no window. The first run builds the project from the Text Commands template, with
   the last layer sized to the commands. Later runs keep training it. It then saves the project and
   exports the program.
3. `comandos.exe --eval test.tsv` reports how many examples it gets right at several confidence
   thresholds. It splits the mistakes into acting wrongly, doing nothing when asked, and the wrong
   command.

Measured with 200,000 sentences and 6 epochs (about 2.5 minutes): 98.9% right on new sentences
with known phrasings (it acts wrongly on 0.8%). On phrasings it never saw it gets 89% right, and
most mistakes are wordings that change the meaning ("como abrir...", "talvez eu abra...
depois"). Such wordings belong in `generate.py`.

## Developer aid

```bash
logievo [project] --size 1600x900 [--select ID] [--train] [--wait MS] --capture shot.png
```

saves the window and quits. Add `--settings PAGE`, `--help PAGE` or `--new-project TEMPLATE` to
capture those windows too.

```bash
logievo --gpu-check out.txt
```

compares graphics-card and processor matrix products.

```bash
logievo --write-icon assets/branding/icon-1024.png
```

refreshes the icon.
