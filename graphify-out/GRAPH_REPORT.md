# Graph Report - Logievo  (2026-10-30)

## Corpus Check
- 83 files · ~74,884 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 10 file(s) not represented in the graph (top: .ttf 4, (none) 3, .cmake 1)

## Summary
- 1802 nodes · 3794 edges · 93 communities (73 shown, 20 thin omitted)
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 539 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- bestyle Icon Enum
- Rgba
- Document Model & Undo
- Worker
- Widgets.cpp
- Node Type Registry
- Editor Window
- Settings Theme Tiles
- Inspector Panel
- Theme Palette Tokens
- DragValue Widget
- Paged Window
- Node Canvas View
- bestyle Helpers & Gallery
- Preferences Pages
- Window Chrome
- Build Targets & README
- Home Screen Logic
- Qt Style Override
- SettingsWindow.cpp
- Canvas Drag & Drop
- Draggable Tile
- path
- Thread Pool
- Shortcut
- Widgets.h
- Graph Preview Rendering
- Runtime.cpp
- generate.py
- Settings Data Model
- Expression Binary Ops
- Home Screen Header
- Graph
- prepareWeights
- Flow Layout
- Node Library Panel
- Dataset.cpp
- Canvas Interaction
- Size-Hint Widgets
- JSON Serialization
- Appearance
- Expression Unary Functions
- Expression Parser
- Project Card
- Responsive Grid
- LabeledRow
- CategoryHeader Fields
- Node Shape Info
- drawLogo
- Template Card
- TrainReport
- Expression Tokens
- App Signals
- Export.cpp
- Tensor
- Expression Program
- saveSettings
- Expression Compiler
- .connect
- .rect
- main
- Canvas Hit Testing
- matmul
- State
- Executor State
- Dual-Number Evaluation
- ChoiceRow
- runner/main.cpp
- State
- Job
- Canvas Drag Modes
- Project Template Record
- App Icon Branding
- CommandModel
- graphify Tooling Rules
- Code Node & Autodiff
- Training Metrics
- Icon Badge
- bestyle Subtree Script
- Live Theme Switching
- Performance & Dev CLI
- Math Nodes & Layers
- Text Translation Routing
- enable
- setTranslator
- FlowLayout::arrange
- RecentProject
- QPaintEvent

## God Nodes (most connected - your core abstractions)
1. `Icon` - 144 edges
2. `QWidget` - 100 edges
3. `NodeCanvas` - 55 edges
4. `Rgba` - 52 edges
5. `Document` - 44 edges
6. `main()` - 43 edges
7. `DragValue` - 38 edges
8. `path` - 37 edges
9. `Palette` - 34 edges
10. `HomeScreen` - 34 edges

## Surprising Connections (you probably didn't know these)
- `mix()` --calls--> `m`  [INFERRED]
  libs/bestyle/src/Theme.cpp → src/app/Gpu.cpp
- `main()` --calls--> `node`  [INFERRED]
  tests/UiTests.cpp → src/ui/NodeCanvas.h
- `headerColor()` --references--> `Rgba`  [EXTRACTED]
  src/ui/EditorWindow.cpp → libs/bestyle/include/bestyle/Draw.h
- `panelColor()` --references--> `Rgba`  [EXTRACTED]
  src/ui/EditorWindow.cpp → libs/bestyle/include/bestyle/Draw.h
- `main()` --references--> `QMimeData`  [INFERRED]
  tests/UiTests.cpp → libs/bestyle/include/bestyle/Widgets.h

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **bestyle rules for live theme changes** — libs_bestyle_readme_color_tokens, libs_bestyle_readme_appearance_changed, libs_bestyle_readme_spacing_metrics, libs_bestyle_readme_translate [EXTRACTED 1.00]
- **Logievo build target layering (core -> ui+bestyle -> exe)** — src_cmakelists_logievo_core, src_cmakelists_logievo_ui, libs_bestyle_cmakelists_bestyle, src_cmakelists_logievo [EXTRACTED 1.00]
- **Training pipeline (Dataset, model nodes, Loss, Trainer)** — readme_dataset_node, readme_primitive_math_nodes, readme_ready_made_layers, readme_loss_node, readme_training [INFERRED 0.85]

## Communities (93 total, 20 thin omitted)

### Community 0 - "bestyle Icon Enum"
Cohesion: 0.02
Nodes (126): Icon, AlignCenter, AlignLeft, AlignRight, ArrowDown, ArrowLeft, ArrowRight, ArrowUp (+118 more)

### Community 1 - "Rgba"
Cohesion: 0.05
Nodes (55): alpha(), blue(), Draw, fromQColor(), green(), Pt, x, y (+47 more)

### Community 2 - "Document Model & Undo"
Cohesion: 0.07
Nodes (12): Document, analysis_, current_, Document::Document(), file, graph, public, report_ (+4 more)

### Community 3 - "Worker"
Cohesion: 0.13
Nodes (6): Worker, cv_, jobs_, mutex_, ready_, surface_

### Community 4 - "Widgets.cpp"
Cohesion: 0.10
Nodes (17): CategoryBody::add(), CategoryBody::CategoryBody(), CategoryHeader::dragLeaveEvent(), DragValue::beginTyping(), DragValue::DragValue(), DragValue::mouseMoveEvent(), DragValue::mousePressEvent(), DragValue::sizeHint() (+9 more)

### Community 5 - "Node Type Registry"
Cohesion: 0.08
Nodes (37): boolParam(), build(), choiceParam(), fileParam(), findNodeType(), findParam(), floatParam(), inputPorts() (+29 more)

### Community 6 - "Editor Window"
Cohesion: 0.08
Nodes (24): Document, EditorWindow, canvas_, doc_, EditorWindow::EditorWindow(), inspector_, library_, menuBar_ (+16 more)

### Community 7 - "Settings Theme Tiles"
Cohesion: 0.07
Nodes (11): LogoButton, Panel, borderBottom, Swatch, chosen_, color_, ThemeTile, accent_ (+3 more)

### Community 8 - "Inspector Panel"
Cohesion: 0.08
Nodes (28): Tone, Accent, Border, Danger, Text, TextDim, TextFaint, toneColor() (+20 more)

### Community 9 - "Theme Palette Tokens"
Cohesion: 0.06
Nodes (32): Palette, accent, bgDeep, bgHover, bgPanel, bgRaised, border, borderWidth (+24 more)

### Community 10 - "DragValue Widget"
Cohesion: 0.06
Nodes (14): DragValue, dragging_, edit_, format_, highlighted_, max_, min_, moved_ (+6 more)

### Community 11 - "Paged Window"
Cohesion: 0.09
Nodes (14): PagedWindow, area_, current_, list_, PagedWindow::PagedWindow(), pages_, public, QLabel (+6 more)

### Community 12 - "Node Canvas View"
Cohesion: 0.06
Nodes (18): Document, NodeCanvas, cursorView_, drag_, framed_, ghost_, hover_, lastView_ (+10 more)

### Community 13 - "bestyle Helpers & Gallery"
Cohesion: 0.15
Nodes (15): main(), section(), install(), translate(), accentButton(), applyTone(), ChoiceRow::ChoiceRow(), DragValue::text() (+7 more)

### Community 14 - "Preferences Pages"
Cohesion: 0.32
Nodes (13): aboutPage(), creditsPage(), learnPage(), licensePage(), shortcutsPage(), addSection(), hintLabel(), pageTitle() (+5 more)

### Community 15 - "Window Chrome"
Cohesion: 0.09
Nodes (8): Chrome, cursorFor(), edgesAt(), Header, title_, window_, Outline, resizable()

### Community 17 - "Build Targets & README"
Cohesion: 0.10
Nodes (26): Inter font (The Inter Project Authors), SIL Open Font License 1.1, cmake/Dependencies.cmake include, LOGIEVO_BUILD_TESTS option, logievo CMake project (C++20, v0.1.0), bestyle static library target, bestyle_gallery example executable, bestyle (bemotion's look as a Qt 6 Widgets library) (+18 more)

### Community 18 - "Home Screen Logic"
Cohesion: 0.16
Nodes (5): bannerCard(), modified(), relativeTime(), sanitizeFileName(), uniqueProjectPath()

### Community 19 - "Qt Style Override"
Cohesion: 0.09
Nodes (6): Style, Style::drawPrimitive(), Style::pixelMetric(), Style::polish(), Style::styleHint(), WheelGuard

### Community 21 - "Canvas Drag & Drop"
Cohesion: 0.14
Nodes (5): QMimeData, choice(), draggedNode(), fmt(), summary()

### Community 22 - "Draggable Tile"
Cohesion: 0.10
Nodes (12): Tile, dimmed_, dragData_, dragPixmap_, highlighted_, icon_, label_, marked_ (+4 more)

### Community 23 - "path"
Cohesion: 0.06
Nodes (32): Application, editor_, home_, stack_, window_, EditorWindow, HomeScreen, MainWindow (+24 more)

### Community 24 - "Thread Pool"
Cohesion: 0.12
Nodes (11): hardwareThreads(), parallelFor(), Pool, cv_, jobs_, mutex_, quit_, workers_ (+3 more)

### Community 25 - "Shortcut"
Cohesion: 0.50
Nodes (4): Shortcut, group, keys, what

### Community 26 - "Widgets.h"
Cohesion: 0.11
Nodes (9): CategoryBody, IconButton, activeColor_, icon_, public, size_, QLineEdit, Switch (+1 more)

### Community 27 - "Graph Preview Rendering"
Cohesion: 0.18
Nodes (9): graphBounds(), inputPort(), nodeRect(), outputPort(), renderPreview(), rows(), wirePath(), drag() (+1 more)

### Community 28 - "Runtime.cpp"
Cohesion: 0.27
Nodes (7): around(), fill(), normAxis(), reshaped(), Trainer::~Trainer(), Shape, transposeLast()

### Community 29 - "generate.py"
Cohesion: 0.12
Nodes (7): Generator, main(), norm(), read_commands(), strip_accents(), write(), main()

### Community 30 - "Settings Data Model"
Cohesion: 0.11
Nodes (17): CustomNode, code, description, inputs, name, Settings, accent, customNodes (+9 more)

### Community 31 - "Expression Binary Ops"
Cohesion: 0.10
Nodes (21): Code, Add, Clamp, Div, Eq, Fn1, Fn2, Ge (+13 more)

### Community 32 - "Home Screen Header"
Cohesion: 0.09
Nodes (20): HomeScreen, columns_, content_, entries_, error_, errorRow_, onCreate, previews_ (+12 more)

### Community 33 - "Graph"
Cohesion: 0.06
Nodes (27): Graph, links, name, nodes, training, Link, fromNode, fromPort (+19 more)

### Community 34 - "prepareWeights"
Cohesion: 0.48
Nodes (4): Batches, makeBatches(), prepareWeights(), weightShapes()

### Community 35 - "Flow Layout"
Cohesion: 0.13
Nodes (4): FlowLayout, items_, FlowLayout::takeAt(), QLayout

### Community 36 - "Node Library Panel"
Cohesion: 0.14
Nodes (11): categoryIcon(), nodeIcon(), NodeLibrary, closed_, list_, listLayout_, onAdd, public (+3 more)

### Community 37 - "Dataset.cpp"
Cohesion: 0.07
Nodes (37): Dataset, classCount, classes, commands, error, features, rows, texts (+29 more)

### Community 40 - "JSON Serialization"
Cohesion: 0.41
Nodes (10): Json, Weights, fromBase64(), fromJson(), loadGraph(), saveGraph(), toBase64(), toJson() (+2 more)

### Community 41 - "Appearance"
Cohesion: 0.22
Nodes (3): Swatches, Appearance, public

### Community 42 - "Expression Unary Functions"
Cohesion: 0.12
Nodes (17): Fn, Abs, Cos, Exp, Floor, Log, Max, Min (+9 more)

### Community 43 - "Expression Parser"
Cohesion: 0.27
Nodes (7): Parser, error_, inputs_, locals_, p_, t_, uses_

### Community 44 - "Project Card"
Cohesion: 0.18
Nodes (6): ProjectCard, menu_, meta_, name_, open_, thumbnail_

### Community 45 - "Responsive Grid"
Cohesion: 0.15
Nodes (6): ResponsiveGrid, cells_, gap_, minCell_, ResponsiveGrid::heightForWidth(), ResponsiveGrid::place()

### Community 46 - "LabeledRow"
Cohesion: 0.15
Nodes (7): LabeledRow, control_, controlMin_, heightOf(), LabeledRow::heightForWidth(), LabeledRow::minimumSizeHint(), LabeledRow::place()

### Community 47 - "CategoryHeader Fields"
Cohesion: 0.11
Nodes (12): CategoryHeader, accepts, closed_, count_, dropped, icon_, menuRequested, name_ (+4 more)

### Community 48 - "Node Shape Info"
Cohesion: 0.12
Nodes (15): Analysis, errors, hasData, hasLoss, nodes, order, totalParams, waiting (+7 more)

### Community 49 - "drawLogo"
Cohesion: 0.10
Nodes (11): BigLogo, KeyCaps, keys_, Part, separator, text, LogoMark, appIcon() (+3 more)

### Community 51 - "TrainReport"
Cohesion: 0.15
Nodes (9): TrainReport, accuracy, epoch, epochs, error, loss, running, LossChart (+1 more)

### Community 52 - "Expression Tokens"
Cohesion: 0.17
Nodes (10): Kind, End, Name, Newline, Number, Symbol, Token, kind (+2 more)

### Community 53 - "App Signals"
Cohesion: 0.18
Nodes (4): NodeCanvas::NodeCanvas(), NodeLibrary::NodeLibrary(), AppSignals, public

### Community 54 - "Export.cpp"
Cohesion: 0.27
Nodes (9): decide(), Decision, command, confidence, exportExecutable(), fail(), fitOutputToClasses(), prepareCommandModel() (+1 more)

### Community 55 - "Tensor"
Cohesion: 0.26
Nodes (5): binary(), Tensor, shape, v, unary()

### Community 56 - "Expression Program"
Cohesion: 0.20
Nodes (8): Expression, kMaxInputs, program_, slots_, Op, arg, code, value

### Community 57 - "saveSettings"
Cohesion: 0.24
Nodes (10): applyLook(), applyPerformance(), saveSettings(), QTimer, openHelp(), changedLook(), interfacePage(), openSettings() (+2 more)

### Community 59 - ".connect"
Cohesion: 0.38
Nodes (3): HomeScreen::HomeScreen(), onOpen, styled()

### Community 60 - ".rect"
Cohesion: 0.18
Nodes (7): toggled, CategoryHeader::mouseReleaseEvent(), CategoryHeader::paintEvent(), ElidedLabel::paintEvent(), paintCategoryHeader(), font, line

### Community 61 - "main"
Cohesion: 0.16
Nodes (10): analyze(), Trainer, fresh_, latest_, mutex_, thread_, add(), gradientsMatch() (+2 more)

### Community 62 - "Canvas Hit Testing"
Cohesion: 0.22
Nodes (9): Hit, kind, node, port, Kind, Input, Node, None (+1 more)

### Community 63 - "matmul"
Cohesion: 0.36
Nodes (6): checkGpu(), broadcastIndex(), broadcastShapes(), elementCount(), matmul(), reduceTo()

### Community 65 - "State"
Cohesion: 0.50
Nodes (4): State, Error, Ok, Waiting

### Community 66 - "Executor State"
Cohesion: 0.15
Nodes (9): Executor, state_, State, cache, derivs, grad, index, out (+1 more)

### Community 67 - "Dual-Number Evaluation"
Cohesion: 0.48
Nodes (5): constant(), Dual, d, v, scaled()

### Community 70 - "ChoiceRow"
Cohesion: 0.29
Nodes (4): ChoiceRow, icon_, label_, public

### Community 71 - "runner/main.cpp"
Cohesion: 0.21
Nodes (15): arguments(), error(), evaluate(), expand(), fromUtf8(), main(), open(), perform() (+7 more)

### Community 72 - "State"
Cohesion: 0.17
Nodes (9): Info, available, device, error, State, on, probed, surface (+1 more)

### Community 73 - "Job"
Cohesion: 0.18
Nodes (10): Job, a, b, c, done, k, m, n (+2 more)

### Community 74 - "Canvas Drag Modes"
Cohesion: 0.33
Nodes (6): Drag, Box, Link, Nodes, None, Pan

### Community 75 - "Project Template Record"
Cohesion: 0.40
Nodes (4): ProjectTemplate, description, id, name

### Community 76 - "App Icon Branding"
Cohesion: 0.50
Nodes (4): Logievo App Icon (1024px), Isometric L Monogram, Monochrome Brand Palette (near-black, white, gray), Rounded-Square Icon Container

### Community 77 - "CommandModel"
Cohesion: 0.18
Nodes (8): CommandModel, analysis, commands, dataset, graph, port, prediction, weights

### Community 80 - "Training Metrics"
Cohesion: 0.67
Nodes (3): Metrics, correct, counted

### Community 88 - "enable"
Cohesion: 0.24
Nodes (8): enable(), enabled(), probe(), shutdown(), info, state, MatmulKernel, setMatmulKernel()

### Community 91 - "RecentProject"
Cohesion: 0.25
Nodes (3): RecentProject, lastOpened, name

### Community 94 - "QPaintEvent"
Cohesion: 0.18
Nodes (6): CategoryBody::paintEvent(), ChoiceRow::paintEvent(), DragValue::paintEvent(), IconButton::paintEvent(), SegmentStrip, Tile::paintEvent()

## Knowledge Gaps
- **529 isolated node(s):** `public`, `x`, `y`, `Text`, `Rectangle` (+524 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 815 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **20 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `QWidget` connect `bestyle Helpers & Gallery` to `Widgets.cpp`, `Editor Window`, `Settings Theme Tiles`, `Inspector Panel`, `DragValue Widget`, `Paged Window`, `Node Canvas View`, `Preferences Pages`, `Window Chrome`, `Core Headers Hub`, `Home Screen Logic`, `Qt Style Override`, `SettingsWindow.cpp`, `Canvas Drag & Drop`, `path`, `Widgets.h`, `Graph Preview Rendering`, `Home Screen Header`, `Node Library Panel`, `Appearance`, `Project Card`, `Responsive Grid`, `LabeledRow`, `CategoryHeader Fields`, `drawLogo`, `TrainReport`, `App Signals`, `saveSettings`, `.connect`, `Icon Badge`, `RecentProject`, `QPaintEvent`?**
  _High betweenness centrality (0.209) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `QWidget` (e.g. with `main()` and `.rebuild()`) actually correct?**
  _`QWidget` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `public`, `x`, `y` to the rest of the system?**
  _529 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `bestyle Icon Enum` be split into smaller, more focused modules?**
  _Cohesion score 0.015873015873015872 - nodes in this community are weakly interconnected._
- **Why does `Icon` connect `bestyle Icon Enum` to `Application.cpp`, `Rgba`, `Node Library Panel`, `ChoiceRow`, `bestyle Helpers & Gallery`, `CategoryHeader Fields`, `Icon Badge`, `Draggable Tile`, `Widgets.h`, `.rect`?**
  _High betweenness centrality (0.163) - this node is a cross-community bridge._
- **Should `Rgba` be split into smaller, more focused modules?**
  _Cohesion score 0.05307709906199954 - nodes in this community are weakly interconnected._
- **Why does `Palette` connect `Theme Palette Tokens` to `Rgba`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._