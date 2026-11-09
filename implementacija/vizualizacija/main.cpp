#include "../src/nbody.h"
#include <windows.h>
#include <windowsx.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
constexpr double dt = 0.002;
constexpr int steps_per_frame = 4;
constexpr int panel_width = 300;
enum Control { count_id = 101, seed_id, distribution_id, backend_id,
    theta_id, start_id, reset_id, fit_id, random_id, status_id };
constexpr const wchar_t* backend_labels[] = {
    L"Direct - osnovna", L"Direct - blokovska", L"Direct - SIMD",
    L"Barnes-Hut - osnovna", L"Barnes-Hut - optimizovana"};
static_assert(std::size(backend_labels) == nbody::backends.size());
constexpr auto default_backend = nbody::Backend::bh_optimized;

// Prozor posjeduje samo svoje stanje i kontrole. Fizika, octree i integrator
// ostaju u nbody_core, istom kodu koji koriste benchmark i numericki testovi.
struct App {
    HWND window = nullptr;
    HFONT font = nullptr, heading_font = nullptr;
    HBRUSH panel_brush = nullptr;
    double dpi_scale = 1, view_x = 0, view_y = 0, view_half = 1.25;
    nbody::Bodies bodies;
    nbody::Acceleration acceleration;
    nbody::Config config;
    nbody::Backend backend = default_backend;
    std::uint64_t steps = 0;
    double simulation_time = 0, compute_ms = 0;
    bool running = false, dirty = false, clustered = false;
};

int px(const App& app, int value) { return int(std::lround(value * app.dpi_scale)); }
HWND control(const App& app, int id) { return GetDlgItem(app.window, id); }

std::wstring text(HWND field) {
    const int length = GetWindowTextLengthW(field);
    std::wstring result(length + 1, L'\0');
    GetWindowTextW(field, result.data(), length + 1);
    result.resize(length);
    return result;
}

std::uint64_t integer(HWND field) {
    const auto value = text(field);
    if (value.empty() || !std::all_of(value.begin(), value.end(), [](wchar_t c) { return c >= L'0' && c <= L'9'; }))
        throw std::invalid_argument("Unesite nenegativan cijeli broj.");
    return std::stoull(value);
}

nbody::Backend selected_backend(const App& app) {
    const auto selected = SendMessageW(control(app, backend_id), CB_GETCURSEL, 0, 0);
    if (selected < 0 || std::size_t(selected) >= nbody::backends.size())
        throw std::invalid_argument("Odaberite implementaciju.");
    return nbody::backends[std::size_t(selected)];
}

void update_controls(const App& app) {
    for (int id : {count_id, seed_id, distribution_id, backend_id, random_id})
        EnableWindow(control(app, id), !app.running);
    const auto backend = selected_backend(app);
    EnableWindow(control(app, theta_id), !app.running && nbody::barnes_hut_backend(backend));
    SetWindowTextW(control(app, start_id), app.running ? L"Pauziraj" : L"Pokreni");
}

void update_status(const App& app) {
    std::wostringstream status;
    status << (app.running ? L"Pokrenuto" : app.dirty ? L"Postavke cekaju primjenu" : L"Pauzirano")
           << L"\nTijela: " << app.bodies.size()
           << L"\nKoraci: " << app.steps << L"   t: " << std::fixed << std::setprecision(3) << app.simulation_time
           << L"\nProracun: " << std::setprecision(2) << app.compute_ms << L" ms/korak";
    SetWindowTextW(control(app, status_id), status.str().c_str());
}

// Zum i centriranje mijenjaju samo kameru. Polozaji tijela i fizicki model
// ostaju isti; pri resetu ponovo uklapamo cijelo pocetno stanje u prikaz.
void fit_view(App& app) {
    if (app.bodies.size() == 0) return;
    const auto [xmin, xmax] = std::minmax_element(app.bodies.x.begin(), app.bodies.x.end());
    const auto [ymin, ymax] = std::minmax_element(app.bodies.y.begin(), app.bodies.y.end());
    app.view_x = (double(*xmin) + *xmax) / 2;
    app.view_y = (double(*ymin) + *ymax) / 2;
    app.view_half = std::max({double(*xmax) - *xmin, double(*ymax) - *ymin, 0.1}) * 0.6;
    InvalidateRect(app.window, nullptr, FALSE);
}

void reset_simulation(App& app) {
    const auto count = integer(control(app, count_id));
    const auto seed = integer(control(app, seed_id));
    if (count == 0 || count > 10000) throw std::invalid_argument("Broj tijela za prikaz mora biti 1..10000.");
    const auto backend = selected_backend(app);
    auto config = app.config;
    if (nbody::barnes_hut_backend(backend)) {
        const auto value = text(control(app, theta_id));
        std::size_t used = 0;
        const double theta = std::stod(value, &used);
        if (used != value.size() || !std::isfinite(theta) || theta < 0 || theta > 1)
            throw std::invalid_argument("Theta mora biti 0..1. Koristite decimalnu tacku.");
        config.theta = nbody::Real(theta);
    }
    const bool clustered = SendMessageW(control(app, distribution_id), CB_GETCURSEL, 0, 0) == 1;
    auto bodies = nbody::initial_state(std::size_t(count), seed, clustered ? "clustered" : "uniform");
    // Ovo je ravanska simulacija: z i vz su nula. Koristimo istu omeksanu
    // Newtonovu gravitaciju kao benchmark; ne uvodimo drugi 2D zakon sile.
    std::fill(bodies.z.begin(), bodies.z.end(), nbody::Real(0));
    nbody::Acceleration acceleration;
    nbody::accelerations(bodies, acceleration, config, backend);
    app.bodies = std::move(bodies);
    app.acceleration = std::move(acceleration);
    app.config = config;
    app.backend = backend;
    app.clustered = clustered;
    app.steps = 0;
    app.simulation_time = app.compute_ms = 0;
    app.running = app.dirty = false;
    update_controls(app);
    update_status(app);
    fit_view(app);
}

// Tajmer radi nekoliko malih, fiksnih leapfrog koraka. Sporo crtanje ne
// povecava dt: usporava prikaz, umjesto da promijeni numericku integraciju.
void advance(App& app) {
    const auto begin = std::chrono::steady_clock::now();
    for (int k = 0; k < steps_per_frame; ++k)
        nbody::step(app.bodies, app.acceleration, app.config, app.backend, dt);
    app.compute_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count() / steps_per_frame;
    for (const auto* values : {&app.bodies.x, &app.bodies.y, &app.bodies.vx, &app.bodies.vy})
        for (auto value : *values)
            if (!std::isfinite(value)) throw std::runtime_error("Simulacija je izasla iz numerickog opsega. Resetujte stanje.");
    app.steps += steps_per_frame;
    app.simulation_time = double(app.steps) * dt;
    update_status(app);
    InvalidateRect(app.window, nullptr, FALSE);
}

// GDI crta samo postojece x/y koordinate. Dvostruki bafer uklanja treperenje;
// crtanje je izvan tajmera proracuna i odvojenog benchmark programa.
void paint(const App& app) {
    PAINTSTRUCT paint_info;
    HDC screen = BeginPaint(app.window, &paint_info);
    RECT client;
    GetClientRect(app.window, &client);
    const int width = client.right, height = client.bottom;
    if (width <= 0 || height <= 0) { EndPaint(app.window, &paint_info); return; }
    HDC buffer = CreateCompatibleDC(screen);
    HBITMAP bitmap = CreateCompatibleBitmap(screen, width, height);
    if (!buffer || !bitmap) {
        if (buffer) DeleteDC(buffer);
        if (bitmap) DeleteObject(bitmap);
        EndPaint(app.window, &paint_info);
        return;
    }
    const auto old_bitmap = SelectObject(buffer, bitmap);
    FillRect(buffer, &client, app.panel_brush);
    RECT canvas = {px(app, panel_width), 0, width, height};
    SelectObject(buffer, GetStockObject(DC_BRUSH));
    SetDCBrushColor(buffer, RGB(11, 19, 32));
    FillRect(buffer, &canvas, HBRUSH(GetStockObject(DC_BRUSH)));
    SelectObject(buffer, app.font);
    SetBkMode(buffer, TRANSPARENT);
    const int saved = SaveDC(buffer);
    IntersectClipRect(buffer, canvas.left + px(app, 12), px(app, 60), width - px(app, 12), height - px(app, 50));
    const double cx = (double(canvas.left) + width) / 2, cy = double(height) / 2;
    const double scale = std::max(1, std::min(width - int(canvas.left), height) - px(app, 100)) / (2 * app.view_half);
    SelectObject(buffer, GetStockObject(DC_PEN));
    SetDCPenColor(buffer, RGB(26, 38, 54));
    for (int k = 1; k < 10; ++k) {
        const int x = canvas.left + (width - canvas.left) * k / 10, y = height * k / 10;
        MoveToEx(buffer, x, 0, nullptr); LineTo(buffer, x, height);
        MoveToEx(buffer, canvas.left, y, nullptr); LineTo(buffer, width, y);
    }
    SelectObject(buffer, GetStockObject(NULL_PEN));
    const int radius = px(app, app.bodies.size() > 1000 ? 2 : 3);
    for (std::size_t i = 0; i < app.bodies.size(); ++i) {
        const double x = cx + (double(app.bodies.x[i]) - app.view_x) * scale;
        const double y = cy - (double(app.bodies.y[i]) - app.view_y) * scale;
        if (!std::isfinite(x) || !std::isfinite(y)) continue;
        if (x < canvas.left - radius || x > width + radius || y < -radius || y > height + radius) continue;
        SetDCBrushColor(buffer, app.clustered && i % 2 ? RGB(255, 177, 103) : RGB(100, 218, 239));
        Ellipse(buffer, int(x) - radius, int(y) - radius, int(x) + radius + 1, int(y) + radius + 1);
    }
    RestoreDC(buffer, saved);
    SetTextColor(buffer, RGB(226, 235, 245));
    RECT title = {canvas.left + px(app, 24), px(app, 20), width - px(app, 20), px(app, 48)};
    DrawTextW(buffer, L"Gravitaciona simulacija | ravan XY", -1, &title, DT_LEFT | DT_SINGLELINE);
    SetTextColor(buffer, RGB(149, 168, 190));
    RECT hint = {canvas.left + px(app, 24), height - px(app, 34), width - px(app, 20), height};
    DrawTextW(buffer, L"Tocak misa: zum  |  Uklopi: centriranje  |  dt = 0.002", -1, &hint, DT_LEFT | DT_SINGLELINE);
    BitBlt(screen, 0, 0, width, height, buffer, 0, 0, SRCCOPY);
    SelectObject(buffer, old_bitmap);
    DeleteObject(bitmap);
    DeleteDC(buffer);
    EndPaint(app.window, &paint_info);
}

HWND child(App& app, const wchar_t* kind, const wchar_t* caption, DWORD style,
           int id, int x, int y, int width, int height) {
    HWND handle = CreateWindowExW(kind == std::wstring(L"EDIT") ? WS_EX_CLIENTEDGE : 0, kind, caption,
        WS_CHILD | WS_VISIBLE | style, px(app, x), px(app, y), px(app, width), px(app, height),
        app.window, reinterpret_cast<HMENU>(INT_PTR(id)), GetModuleHandleW(nullptr), nullptr);
    if (!handle) throw std::runtime_error("Nije moguce napraviti kontrolu prozora.");
    SendMessageW(handle, WM_SETFONT, WPARAM(app.font), TRUE);
    return handle;
}

// Kontrole su na lijevoj strani, a desno ostaje samo platno. Njihovi ID-jevi
// povezuju korisnicki unos s konfiguracijom pri narednom pokretanju ili resetu.
void create_controls(App& app) {
    const auto title = child(app, L"STATIC", L"N-body 2D", 0, 0, 24, 20, 250, 32);
    SendMessageW(title, WM_SETFONT, WPARAM(app.heading_font), TRUE);
    child(app, L"STATIC", L"CPU simulacija gravitacije", 0, 0, 24, 54, 250, 22);
    const wchar_t* labels[] = {L"Broj tijela (1-10000)", L"Seed pocetnog stanja", L"Pocetni raspored",
        L"Implementacija", L"Theta (Barnes-Hut)"};
    for (int k = 0; k < 5; ++k) child(app, L"STATIC", labels[k], 0, 0, 24, 84 + k * 64, 252, 22);
    child(app, L"EDIT", L"512", ES_NUMBER | WS_TABSTOP, count_id, 24, 106, 252, 30);
    child(app, L"EDIT", L"42", ES_NUMBER | WS_TABSTOP, seed_id, 24, 170, 186, 30);
    child(app, L"BUTTON", L"Novi", WS_TABSTOP, random_id, 218, 170, 58, 30);
    const auto distribution = child(app, L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_VSCROLL | WS_TABSTOP, distribution_id, 24, 234, 252, 180);
    for (auto label : {L"Ravnomjerno", L"Dva klastera"}) SendMessageW(distribution, CB_ADDSTRING, 0, LPARAM(label));
    SendMessageW(distribution, CB_SETCURSEL, 0, 0);
    const auto backend = child(app, L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_VSCROLL | WS_TABSTOP, backend_id, 24, 298, 252, 300);
    for (auto label : backend_labels) SendMessageW(backend, CB_ADDSTRING, 0, LPARAM(label));
    const auto index = std::find(nbody::backends.begin(), nbody::backends.end(), default_backend);
    SendMessageW(backend, CB_SETCURSEL, index - nbody::backends.begin(), 0);
    child(app, L"EDIT", L"0.5", WS_TABSTOP, theta_id, 24, 362, 252, 30);
    child(app, L"BUTTON", L"Pokreni", BS_DEFPUSHBUTTON | WS_TABSTOP, start_id, 24, 420, 252, 38);
    child(app, L"BUTTON", L"Resetuj", WS_TABSTOP, reset_id, 24, 470, 122, 34);
    child(app, L"BUTTON", L"Uklopi prikaz", WS_TABSTOP, fit_id, 154, 470, 122, 34);
    child(app, L"STATIC", L"Isti seed daje isti raspored.\nPromjene: Pokreni ili Resetuj.", 0, 0, 24, 522, 252, 46);
    child(app, L"STATIC", L"", 0, status_id, 24, 584, 252, 96);
}

// Windows ovdje salje klikove, promjene polja, tajmer i zahtjeve za crtanje.
LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
    auto* app = reinterpret_cast<App*>(GetWindowLongPtrW(window, GWLP_USERDATA));
    if (message == WM_NCCREATE) {
        app = static_cast<App*>(reinterpret_cast<CREATESTRUCTW*>(lparam)->lpCreateParams);
        app->window = window;
        SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(app));
    }
    if (!app) return DefWindowProcW(window, message, wparam, lparam);
    try {
        switch (message) {
        case WM_CREATE:
            app->dpi_scale = GetDpiForWindow(window) / 96.0;
            app->font = CreateFontW(-px(*app, 14), 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
            app->heading_font = CreateFontW(-px(*app, 25), 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
            app->panel_brush = CreateSolidBrush(RGB(245, 247, 250));
            create_controls(*app);
            reset_simulation(*app);
            if (!SetTimer(window, 1, 16, nullptr)) throw std::runtime_error("Nije moguce pokrenuti tajmer.");
            return 0;
        case WM_COMMAND:
            if (HIWORD(wparam) == BN_CLICKED && LOWORD(wparam) == start_id) {
                if (!app->running && app->dirty) reset_simulation(*app);
                app->running = !app->running;
                update_controls(*app);
                update_status(*app);
            } else if (HIWORD(wparam) == BN_CLICKED && LOWORD(wparam) == reset_id) {
                reset_simulation(*app);
            } else if (HIWORD(wparam) == BN_CLICKED && LOWORD(wparam) == fit_id) {
                fit_view(*app);
            } else if (HIWORD(wparam) == BN_CLICKED && LOWORD(wparam) == random_id) {
                std::random_device random;
                const auto seed = (std::uint64_t(random()) << 32) | random();
                SetWindowTextW(control(*app, seed_id), std::to_wstring(seed).c_str());
            } else if (LOWORD(wparam) >= count_id && LOWORD(wparam) <= theta_id &&
                       (HIWORD(wparam) == EN_CHANGE || HIWORD(wparam) == CBN_SELCHANGE)) {
                app->dirty = true;
                if (control(*app, start_id)) { update_controls(*app); update_status(*app); }
            }
            return 0;
        case WM_TIMER:
            if (app->running) advance(*app);
            return 0;
        case WM_PAINT: paint(*app); return 0;
        case WM_ERASEBKGND: return 1;
        case WM_CTLCOLORSTATIC:
            SetTextColor(HDC(wparam), RGB(34, 48, 66));
            SetBkColor(HDC(wparam), RGB(245, 247, 250));
            return LRESULT(app->panel_brush);
        case WM_MOUSEWHEEL: {
            POINT point = {GET_X_LPARAM(lparam), GET_Y_LPARAM(lparam)};
            ScreenToClient(window, &point);
            if (point.x >= px(*app, panel_width)) {
                app->view_half = std::clamp(app->view_half * std::pow(0.85, GET_WHEEL_DELTA_WPARAM(wparam) / 120.0), 1e-5, 1e6);
                InvalidateRect(window, nullptr, FALSE);
            }
            return 0;
        }
        case WM_SIZE: InvalidateRect(window, nullptr, FALSE); return 0;
        case WM_GETMINMAXINFO: {
            auto* limits = reinterpret_cast<MINMAXINFO*>(lparam);
            limits->ptMinTrackSize = {px(*app, 900), px(*app, 820)};
            return 0;
        }
        case WM_DESTROY: KillTimer(window, 1); PostQuitMessage(0); return 0;
        }
    } catch (const std::exception& error) {
        app->running = false;
        if (control(*app, start_id)) { update_controls(*app); update_status(*app); }
        MessageBoxA(window, error.what(), "N-body - provjerite postavke", MB_OK | MB_ICONWARNING);
        if (message == WM_CREATE) return -1;
        return 0;
    }
    return DefWindowProcW(window, message, wparam, lparam);
}
} // namespace

int WINAPI WinMain(HINSTANCE instance, HINSTANCE, LPSTR, int show) {
    SetProcessDPIAware();
    App app;
    WNDCLASSW window_class{};
    window_class.lpfnWndProc = window_proc;
    window_class.hInstance = instance;
    window_class.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    window_class.lpszClassName = L"NBodySimulation2D";
    if (!RegisterClassW(&window_class)) return 1;
    HWND window = CreateWindowExW(0, window_class.lpszClassName, L"N-body CPU - 2D simulacija",
        WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN, CW_USEDEFAULT, CW_USEDEFAULT, 1120, 850,
        nullptr, nullptr, instance, &app);
    if (!window) return 1;
    ShowWindow(window, show);
    MSG message{};
    int result;
    while ((result = GetMessageW(&message, nullptr, 0, 0)) > 0) {
        if (!IsDialogMessageW(window, &message)) { TranslateMessage(&message); DispatchMessageW(&message); }
    }
    DeleteObject(app.font);
    DeleteObject(app.heading_font);
    DeleteObject(app.panel_brush);
    return result == -1 ? 1 : int(message.wParam);
}
