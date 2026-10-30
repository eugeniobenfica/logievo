#include "core/Graph.h"

#include "core/NodeTypes.h"

#include <algorithm>
#include <cctype>

namespace logievo {

Node* Graph::find(int id) {
    for (Node& n : nodes)
        if (n.id == id) return &n;
    return nullptr;
}

const Node* Graph::find(int id) const { return const_cast<Graph*>(this)->find(id); }

int Graph::nextId() const {
    int id = 0;
    for (const Node& n : nodes) id = std::max(id, n.id);
    return id + 1;
}

std::string Graph::uniqueName(const std::string& base) const {
    // "Dense" -> "dense", then "dense_2", "dense_3"...: names that read well in code.
    std::string stem;
    for (char c : base) {
        if (std::isalnum(static_cast<unsigned char>(c))) stem += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        else if (!stem.empty() && stem.back() != '_') stem += '_';
    }
    while (!stem.empty() && stem.back() == '_') stem.pop_back();
    if (stem.empty() || std::isdigit(static_cast<unsigned char>(stem[0]))) stem = "node" + stem;
    auto taken = [&](const std::string& n) { return std::any_of(nodes.begin(), nodes.end(), [&](const Node& x) { return x.name == n; }); };
    if (!taken(stem)) return stem;
    for (int i = 2;; ++i)
        if (const std::string n = stem + "_" + std::to_string(i); !taken(n)) return n;
}

Node& Graph::add(const std::string& type, float x, float y) {
    Node n;
    n.id = nextId();
    n.type = type;
    const NodeType* t = findNodeType(type);
    n.name = uniqueName(t ? t->name : type);
    n.x = x;
    n.y = y;
    nodes.push_back(std::move(n));
    return nodes.back();
}

void Graph::remove(int id) {
    std::erase_if(links, [id](const Link& l) { return l.fromNode == id || l.toNode == id; });
    std::erase_if(nodes, [id](const Node& n) { return n.id == id; });
}

const Link* Graph::linkInto(int toNode, int toPort) const {
    for (const Link& l : links)
        if (l.toNode == toNode && l.toPort == toPort) return &l;
    return nullptr;
}

bool Graph::reaches(int from, int to) const {
    // Depth-first along the links, from `from` forward.
    std::vector<int> stack{from}, seen;
    while (!stack.empty()) {
        const int n = stack.back();
        stack.pop_back();
        if (n == to) return true;
        if (std::find(seen.begin(), seen.end(), n) != seen.end()) continue;
        seen.push_back(n);
        for (const Link& l : links)
            if (l.fromNode == n) stack.push_back(l.toNode);
    }
    return false;
}

bool Graph::connect(int fromNode, int fromPort, int toNode, int toPort) {
    const Node* a = find(fromNode);
    const Node* b = find(toNode);
    if (!a || !b || fromNode == toNode) return false;
    const NodeType* ta = findNodeType(a->type);
    const NodeType* tb = findNodeType(b->type);
    if (!ta || !tb || fromPort < 0 || fromPort >= static_cast<int>(ta->outputs.size()) || toPort < 0 ||
        toPort >= static_cast<int>(inputPorts(*b).size()))
        return false;
    if (reaches(toNode, fromNode)) return false; // a loop
    disconnect(toNode, toPort);
    links.push_back({fromNode, fromPort, toNode, toPort});
    return true;
}

void Graph::disconnect(int toNode, int toPort) {
    std::erase_if(links, [&](const Link& l) { return l.toNode == toNode && l.toPort == toPort; });
}

Json Graph::param(const Node& node, const std::string& id) const {
    if (auto it = node.params.find(id); it != node.params.end()) return it->second;
    if (const NodeType* t = findNodeType(node.type))
        if (const ParamSpec* p = findParam(*t, id)) {
            if (p->kind == ParamKind::Shape || p->kind == ParamKind::File || p->kind == ParamKind::Code || p->kind == ParamKind::Text) return p->defText;
            if (p->kind == ParamKind::Bool) return p->defNumber != 0.0;
            return p->defNumber;
        }
    return nullptr;
}

double Graph::number(const Node& node, const std::string& id) const {
    const Json v = param(node, id);
    if (v.is_number()) return v.get<double>();
    if (v.is_boolean()) return v.get<bool>() ? 1.0 : 0.0;
    return 0.0;
}

std::string Graph::text(const Node& node, const std::string& id) const {
    const Json v = param(node, id);
    return v.is_string() ? v.get<std::string>() : std::string();
}

} // namespace logievo
