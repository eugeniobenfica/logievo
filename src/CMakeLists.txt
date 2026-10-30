# The model and the engine, in plain C++ (no Qt): graphs, node types, shape inference, tensors,
# datasets, forward and backward passes, training, files.
add_library(logievo_core STATIC
    core/Graph.cpp
    core/NodeTypes.cpp
    core/Analysis.cpp
    core/Tensor.cpp
    core/Parallel.cpp
    core/Expression.cpp
    core/Dataset.cpp
    core/Runtime.cpp
    core/Serialization.cpp
    core/Templates.cpp
    core/Export.cpp)
target_include_directories(logievo_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(logievo_core PUBLIC nlohmann_json::nlohmann_json)

# The interface: home screen and node editor (Qt 6 Widgets, bestyle).
add_library(logievo_ui STATIC
    app/Application.cpp
    app/Settings.cpp
    app/Platform.cpp
    app/Preferences.cpp
    app/Gpu.cpp
    ui/Logo.cpp
    ui/Icons.cpp
    ui/HomeScreen.cpp
    ui/Document.cpp
    ui/EditorWindow.cpp
    ui/NodeLibrary.cpp
    ui/NodeCanvas.cpp
    ui/Inspector.cpp
    ui/GraphPreview.cpp
    ui/WindowChrome.cpp
    ui/PagedWindow.cpp
    ui/SettingsWindow.cpp
    ui/HelpWindow.cpp
    ui/Windows.h) # signals with no .cpp of their own: listed for moc
set_target_properties(logievo_ui PROPERTIES AUTOMOC ON)
target_compile_definitions(logievo_ui PRIVATE LOGIEVO_VERSION="${PROJECT_VERSION}")
target_link_libraries(logievo_ui PUBLIC logievo_core bestyle Qt6::Widgets)

# The runner: a text-commands model on the command line. Export copies it with the model
# appended, so it is linked statically: the exported program runs on its own.
add_executable(logievo_run runner/main.cpp)
target_link_libraries(logievo_run PRIVATE logievo_core)
if(WIN32)
    target_link_libraries(logievo_run PRIVATE shell32)
endif()
if(MINGW)
    target_link_options(logievo_run PRIVATE -static)
endif()

# The app.
add_executable(logievo WIN32 main.cpp)
target_link_libraries(logievo PRIVATE logievo_ui)
add_dependencies(logievo logievo_run) # Export needs it next to the app

# Runtime assets (the Inter font) next to the executable.
add_custom_command(TARGET logievo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/assets $<TARGET_FILE_DIR:logievo>/assets)
