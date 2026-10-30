# Third-party dependencies.
#
# Installed on the system (MSYS2 UCRT64 on Windows, the package manager elsewhere):
#   Qt 6        - user interface (Widgets)
#
# Fetched at configure time and pinned to exact versions:
#   nlohmann    - JSON for the project files and settings (and bestyle's themes)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# --- Qt ------------------------------------------------------------------------
find_package(Qt6 6.4 REQUIRED COMPONENTS Core Gui Widgets)
qt_standard_project_setup()

# --- nlohmann/json -----------------------------------------------------------
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(json)
