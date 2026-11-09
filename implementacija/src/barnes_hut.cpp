#include "nbody.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace nbody {
namespace {

// Čvorovi su u jednom nizu, redom roditelj pa njegova podstabla. Raspon
// begin/end opisuje članove u order, a next preskače cijelo podstablo.
struct Node {
    std::array<std::size_t, 8> children{};
    std::size_t begin = 0, end = 0, next = 0;
    double width = 0;
    Real mass = 0, x = 0, y = 0, z = 0;
    bool leaf = true;
};

// order cuva prostorni redoslijed izvornih indeksa, rank daje obratno mapiranje.
// scratch sluzi samo podjeli pri gradnji; izlaz uvijek koristi izvorni indeks.
struct Tree {
    std::vector<Node> nodes;
    std::vector<std::size_t> order, rank, scratch;
    std::size_t depth = 0;
};

// Svaka koordinata bira donju ili gornju polovinu kocke. Tri bita daju jedan
// od osam oktanata; tacke na srednjoj ravni dosljedno idu u gornju polovinu.
int octant(const Bodies& bodies, std::size_t i, const std::array<double, 3>& center) {
    return int(bodies.x[i] >= center[0]) | (int(bodies.y[i] >= center[1]) << 1)
         | (int(bodies.z[i] >= center[2]) << 2);
}

std::size_t build_node(Tree& tree, const Bodies& bodies, const Config& config,
                       std::size_t begin, std::size_t end,
                       const std::array<double, 3>& center, double half, std::size_t depth) {
    const auto index = tree.nodes.size();
    tree.nodes.emplace_back();
    tree.depth = std::max(tree.depth, depth);
    Node node;
    node.begin = begin;
    node.end = end;
    node.width = 2 * half;

    // Masa i centar mase pripremaju se u double preciznosti, a zatim čuvaju
    // u Real kao ostali fizički podaci. Pomjerena suma smanjuje gubitak cifara.
    const auto anchor = tree.order[begin];
    double mass = 0, x = 0, y = 0, z = 0;
    bool coincident = true;
    for (auto k = begin; k < end; ++k) {
        const auto i = tree.order[k];
        const double m = bodies.mass[i];
        mass += m;
        x += m * (double(bodies.x[i]) - bodies.x[anchor]);
        y += m * (double(bodies.y[i]) - bodies.y[anchor]);
        z += m * (double(bodies.z[i]) - bodies.z[anchor]);
        coincident = coincident && bodies.x[i] == bodies.x[anchor]
            && bodies.y[i] == bodies.y[anchor] && bodies.z[i] == bodies.z[anchor];
    }
    node.mass = Real(mass);
    node.x = Real(double(bodies.x[anchor]) + x / mass);
    node.y = Real(double(bodies.y[anchor]) + y / mass);
    node.z = Real(double(bodies.z[anchor]) + z / mass);
    if (!std::isfinite(node.mass) || !std::isfinite(node.x) ||
        !std::isfinite(node.y) || !std::isfinite(node.z)) {
        throw std::invalid_argument("Masa ili centar mase stabla prelazi numericki opseg");
    }

    // Svaki izvor ulazi u tačno jedan oktant. Jednake pozicije i maksimalna
    // dubina završavaju kao višestruki list, umjesto beskonačnog dijeljenja.
    if (end - begin > config.leaf_capacity && depth < config.max_depth && !coincident && half > 0) {
        std::array<std::size_t, 8> counts{};
        for (auto k = begin; k < end; ++k) ++counts[octant(bodies, tree.order[k], center)];
        std::array<std::size_t, 9> offsets{};
        offsets[0] = begin;
        for (int child = 0; child < 8; ++child) offsets[child + 1] = offsets[child] + counts[child];
        auto cursor = offsets;
        for (auto k = begin; k < end; ++k) {
            const auto i = tree.order[k];
            tree.scratch[cursor[octant(bodies, i, center)]++] = i;
        }
        std::copy(tree.scratch.begin() + begin, tree.scratch.begin() + end, tree.order.begin() + begin);
        node.leaf = false;
        const double child_half = half / 2;
        for (int child = 0; child < 8; ++child) {
            if (counts[child] == 0) continue;
            const std::array<double, 3> child_center = {
                center[0] + (child & 1 ? child_half : -child_half),
                center[1] + (child & 2 ? child_half : -child_half),
                center[2] + (child & 4 ? child_half : -child_half)};
            node.children[child] = build_node(tree, bodies, config, offsets[child], offsets[child + 1],
                                               child_center, child_half, depth + 1);
        }
    }
    node.next = tree.nodes.size();
    tree.nodes[index] = node;
    return index;
}

// Korijenska kocka obuhvata sve tri ose, pa se gradi zajednicko stablo.
// Mala ekspanzija granice cuva krajnje tacke, a rank se popunjava nakon podjele.
Tree build_tree(const Bodies& bodies, const Config& config) {
    Tree tree;
    tree.order.resize(bodies.size());
    tree.rank.resize(bodies.size());
    tree.scratch.resize(bodies.size());
    std::iota(tree.order.begin(), tree.order.end(), 0);
    if (bodies.size() == 0) return tree;
    std::array<double, 3> center{};
    double half = 0;
    const std::array<const std::vector<Real>*, 3> axes = {&bodies.x, &bodies.y, &bodies.z};
    for (int axis = 0; axis < 3; ++axis) {
        const auto [minimum, maximum] = std::minmax_element(axes[axis]->begin(), axes[axis]->end());
        center[axis] = double(*minimum) / 2 + double(*maximum) / 2;
        half = std::max(half, std::max(std::abs(double(*minimum) - center[axis]),
                                     std::abs(double(*maximum) - center[axis])));
    }
    if (half > 0) half = std::nextafter(half, std::numeric_limits<double>::infinity());
    if (!std::isfinite(2 * half)) throw std::invalid_argument("Prostorni opseg stabla je prevelik");
    build_node(tree, bodies, config, 0, bodies.size(), center, half, 0);
    for (std::size_t k = 0; k < tree.order.size(); ++k) tree.rank[tree.order[k]] = k;
    return tree;
}

bool accept(const Node& node, const Tree& tree, const Bodies& bodies, std::size_t target, Real theta) {
    // Čvor koji sadrži cilj uvijek otvaramo: njegova ukupna masa uključuje
    // i cilj. Članstvo se provjerava indeksom, bez nepouzdanih poređenja granica.
    const auto rank = tree.rank[target];
    if (theta == 0 || (rank >= node.begin && rank < node.end)) return false;
    const double dx = double(node.x) - bodies.x[target];
    const double dy = double(node.y) - bodies.y[target];
    const double dz = double(node.z) - bodies.z[target];
    // Ovo je width/distance < theta bez dodatnog korijena. Geometrijska
    // udaljenost bira aproksimaciju; epsilon se koristi tek u fizickoj sili.
    return node.width * node.width < double(theta) * theta * (dx * dx + dy * dy + dz * dz);
}

// Ista gravitaciona formula vazi za pojedinacni izvor i prihvacenu grupu.
// Grupa daje ukupnu masu i centar mase, uz isti epsilon kao Direct proracun.
void add_force(const Bodies& bodies, std::size_t target, Real x, Real y, Real z,
               Real mass, const Config& config, Real& ax, Real& ay, Real& az) {
    const Real dx = x - bodies.x[target], dy = y - bodies.y[target], dz = z - bodies.z[target];
    const Real inverse = Real(1) / std::sqrt(dx * dx + dy * dy + dz * dz + config.epsilon * config.epsilon);
    const Real factor = config.gravity * mass * inverse * inverse * inverse;
    ax += dx * factor;
    ay += dy * factor;
    az += dz * factor;
}

// U listu nema grupne aproksimacije: sabiraju se pojedinacna djelovanja.
// Preskacemo vlastiti indeks, ali razlicita tijela na istoj poziciji ostaju izvori.
void leaf_force(const Node& node, const Tree& tree, const Bodies& bodies, std::size_t target,
                const Config& config, Real& ax, Real& ay, Real& az) {
    for (auto k = node.begin; k < node.end; ++k) {
        const auto source = tree.order[k];
        if (source == target) continue;
        add_force(bodies, target, bodies.x[source], bodies.y[source], bodies.z[source],
                  bodies.mass[source], config, ax, ay, az);
    }
}

// Osnovni obilazak prati osam djece rekurzivno. Listovi se računaju direktno,
// a prihvaćeni udaljeni čvor daje jedno djelovanje iz svog centra mase.
void recursive_force(std::size_t index, const Tree& tree, const Bodies& bodies, std::size_t target,
                     const Config& config, Real& ax, Real& ay, Real& az) {
    const auto& node = tree.nodes[index];
    if (node.leaf) {
        leaf_force(node, tree, bodies, target, config, ax, ay, az);
    } else if (accept(node, tree, bodies, target, config.theta)) {
        add_force(bodies, target, node.x, node.y, node.z, node.mass, config, ax, ay, az);
    } else {
        for (auto child : node.children) {
            if (child != 0) recursive_force(child, tree, bodies, target, config, ax, ay, az);
        }
    }
}

// Optimizovani obilazak ne koristi rekurziju ni pomoćni stek. Otvaranje ide
// na sljedeći čvor; prihvaćanje/list ide na next, iza cijelog podstabla.
void iterative_force(const Tree& tree, const Bodies& bodies, std::size_t target,
                     const Config& config, Real& ax, Real& ay, Real& az) {
    std::size_t index = 0;
    while (index < tree.nodes.size()) {
        const auto& node = tree.nodes[index];
        if (node.leaf) {
            leaf_force(node, tree, bodies, target, config, ax, ay, az);
            index = node.next;
        } else if (accept(node, tree, bodies, target, config.theta)) {
            add_force(bodies, target, node.x, node.y, node.z, node.mass, config, ax, ay, az);
            index = node.next;
        } else {
            ++index;
        }
    }
}

} // namespace

// Svaki poziv ponovo gradi stablo trenutnih polozaja. Obje izvedbe dijele
// gradnju i kriterij otvaranja, a razlikuju se načinom obilaska.
void barnes_hut(const Bodies& bodies, Acceleration& acceleration, const Config& config,
                Backend backend, SolverTiming* timing) {
    const auto start = std::chrono::steady_clock::now();
    const auto tree = build_tree(bodies, config);
    const auto built = std::chrono::steady_clock::now();
    if (backend == Backend::bh_sequential) {
        for (std::size_t i = 0; i < bodies.size(); ++i) {
            Real ax = 0, ay = 0, az = 0;
            recursive_force(0, tree, bodies, i, config, ax, ay, az);
            acceleration.x[i] = ax;
            acceleration.y[i] = ay;
            acceleration.z[i] = az;
        }
    } else {
        // Bliski ciljevi obicno citaju slicne cvorove, sto pomaze kesu.
        // Upis preko izvornog i zadrzava rezultat u istom poretku kao Bodies.
        for (auto i : tree.order) {
            Real ax = 0, ay = 0, az = 0;
            iterative_force(tree, bodies, i, config, ax, ay, az);
            acceleration.x[i] = ax;
            acceleration.y[i] = ay;
            acceleration.z[i] = az;
        }
    }
    const auto finished = std::chrono::steady_clock::now();
    if (timing) {
        timing->tree_seconds = std::chrono::duration<double>(built - start).count();
        timing->force_seconds = std::chrono::duration<double>(finished - built).count();
        timing->tree_nodes = tree.nodes.size();
        timing->tree_depth = tree.depth;
    }
}

} // namespace nbody
