#include "ui/Icons.h"

#include <map>

namespace logievo::ui {

using bestyle::Icon;

Icon nodeIcon(const std::string& type) {
    static const std::map<std::string, Icon> icons{
        {"dataset", Icon::Table},       {"parameter", Icon::Sliders},  {"constant", Icon::Counter},  {"add", Icon::Plus},
        {"subtract", Icon::Minimize},   {"multiply", Icon::Close},     {"divide", Icon::Mask},       {"matmul", Icon::Compound},
        {"sum", Icon::Levels},          {"mean", Icon::Meter},         {"power", Icon::Burst},       {"exp", Icon::Expression},
        {"log", Icon::Graph},           {"relu", Icon::LineChart},     {"leakyrelu", Icon::Zigzag},  {"gelu", Icon::Wave},
        {"sigmoid", Icon::Ease},        {"tanh", Icon::Heartbeat},     {"softmax", Icon::PieChart},  {"dense", Icon::Cube},
        {"conv2d", Icon::Image},        {"embedding", Icon::Text}, {"rnn", Icon::Loop},     {"layernorm", Icon::Scopes},  {"dropout", Icon::Fade},
        {"reshape", Icon::Crop},        {"flatten", Icon::Ungroup},    {"transpose", Icon::Restore}, {"concat", Icon::Duplicate},
        {"maxpool2d", Icon::ZoomOut},   {"globalpool", Icon::FitView}, {"loss", Icon::Track},         {"custom", Icon::Edit},
    };
    auto it = icons.find(type);
    return it == icons.end() ? Icon::Cube : it->second;
}

Icon categoryIcon(const std::string& category) {
    static const std::map<std::string, Icon> icons{
        {"Data", Icon::Table},   {"Math", Icon::Expression}, {"Activations", Icon::LineChart},
        {"Layers", Icon::Cube},  {"Shape", Icon::Crop},      {"Custom", Icon::Edit},          {"Training", Icon::Track},
    };
    auto it = icons.find(category);
    return it == icons.end() ? Icon::Folder : it->second;
}

} // namespace logievo::ui
