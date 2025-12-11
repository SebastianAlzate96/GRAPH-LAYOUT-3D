// ============================================================================
// mini_fr_3d_xyz.cpp — FR en 3D + render con GLUT
// Linux (WSL Ubuntu): sudo apt-get install freeglut3-dev
// Compilar: g++ mini_fr_3d_xyz.cpp -o mini_fr_3d_xyz -lGL -lGLU -lglut && ./mini_fr_3d_xyz
// ============================================================================

#define _USE_MATH_DEFINES
#include <GL/glut.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <cstring>  // strlen
#include <cctype>   // std::isalpha
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <tuple>
#include <string>
using namespace std;

// ============================================================================
// Tipos y utilidades vectoriales
// ============================================================================
struct Vec3 { float x,y,z; };
struct Edge{ int u,v; };

static inline Vec3  add(Vec3 a, Vec3 b){ return {a.x+b.x, a.y+b.y, a.z+b.z}; }
static inline Vec3  sub(Vec3 a, Vec3 b){ return {a.x-b.x, a.y-b.y, a.z-b.z}; }
static inline Vec3  mul(Vec3 a, float s){ return {a.x*s, a.y*s, a.z*s}; }
static inline float dot(Vec3 a, Vec3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline float len(Vec3 a){ return sqrtf(max(1e-12f, dot(a,a))); }
static inline Vec3  norm(Vec3 a){ float L=len(a); return {a.x/L, a.y/L, a.z/L}; }
static inline float clampf(float v,float a,float b){ return v<a?a:(v>b?b:v); }
static inline float dist2(const Vec3& a, const Vec3& b){
    float dx=a.x-b.x, dy=a.y-b.y, dz=a.z-b.z; return dx*dx+dy*dy+dz*dz;
}

// ============================================================================
/* Escena / globals */
// ============================================================================
static int   N   = 48;
static float T   = 0.12f;
static bool  RUN = true;
static bool  COOL= true;
static int   iters_per_frame = 2;        // preset para fluidez
static bool  AUTO_ROT = false;           // auto-rotación del MODELO

static const float T_MIN  = 1e-4f;
static const float T_KICK = 0.03f;
static const float T_MAX  = 1.0f;

static vector<Vec3> P;       // posiciones físicas
static vector<Vec3> D;       // desplazamientos
static vector<Edge> E;       // aristas
static vector<Vec3> P_prev;  // posiciones previas (Δ)
static vector<int>  DEG;     // grado por nodo
static int          DEG_MAX = 1;

static float node_radius = 0.005f;
static int   node_slices = 14, node_stacks = 14;
static bool  USE_LIGHT = true;

static bool g_want_snap = false;

// === HUD extra ===
static bool  HUD_HELP = true;          // panel de ayuda on/off
static int   HUD_PAD  = 8;             // padding del recuadro
static float HUD_BG[4] = {1,1,1,0.80f}; // fondo semi-transparente
static float HUD_FG[3] = {0,0,0};       // texto negro

// ============================================================================
// Coloreo por modo: grado o comunidades (k-means)
// ============================================================================
enum ColorMode { BY_DEGREE, BY_COMMUNITY };
static ColorMode COLOR_MODE = BY_DEGREE;

static int K_COMM = 4;                 // K de k-means
static std::vector<int>  COMM;         // etiqueta de comunidad por nodo
static std::vector<Vec3> CENT;         // centroides de k-means
static int C_COMM = 0;                 // # de comunidades detectadas
static std::mt19937 RNG_KM(1337);      // RNG estable

// Paleta fija para comunidades (12)
static const float PALETTE[12][3] = {
    {0.93f,0.33f,0.31f}, {0.30f,0.69f,0.31f}, {0.30f,0.54f,0.85f},
    {0.98f,0.77f,0.18f}, {0.56f,0.27f,0.68f}, {0.00f,0.74f,0.83f},
    {0.96f,0.26f,0.76f}, {0.55f,0.34f,0.29f}, {0.37f,0.62f,0.63f},
    {0.76f,0.49f,0.26f}, {0.24f,0.24f,0.24f}, {0.56f,0.56f,0.56f}
};
static inline void community_color(int cid, float& r,float& g,float& b){
    const int M = 12;
    const float* c = PALETTE[((cid % M)+M)%M];
    r=c[0]; g=c[1]; b=c[2];
}

// ============================================================================
// ======= Modo de fuerzas =======
// ============================================================================
// ======= Modo de fuerzas =======
enum ForceModel { FM_FR, FM_LINLOG, FM_SPRINGELEC };
static ForceModel FORCE_MODE = FM_FR;

// Parámetros del Spring-Electrical
static float SE_Cr = 1.0f;
static float SE_Ca = 1.0f;
static float SE_L  = 0.9f;

// Cooldown de enfriamiento
static int heat_cooldown_frames = 0;

// ============================================================================
// HUD / métricas
// ============================================================================
static int   g_win_w = 1100, g_win_h = 850;
static float g_fps = 0.0f;
static int   g_frames = 0;
static int   g_prevTime = 0;     // ms (GLUT_ELAPSED_TIME)
static float g_lastDelta = 0.0f; // Δ medio última iteración

// ============================================================================
// Cámara (estática). Rotamos el MODELO, no la cámara.
// ============================================================================
static float cam_radius = 3.5f;
static float model_yaw   = 0.0f;  // rotación Y
static float model_pitch = 15.0f; // rotación X

// ============================================================================
// Suavizado temporal (EMA) sobre la posición para el render
// ============================================================================
static bool  USE_SMOOTH = true;
static float ALPHA      = 0.25f;       // 0..1
static vector<Vec3> P_smooth;

// ============================================================================
// Resalte última arista añadida y anti-duplicados
// ============================================================================
static Edge LAST_ADDED = {-1,-1};

static unordered_set<long long> ESET;
static inline long long ek(int u,int v){
    if(u>v) std::swap(u,v);
    return ( (long long)u << 32 ) | (unsigned long long)v;
}
static inline bool has_edge(int u,int v){
    return ESET.find(ek(u,v)) != ESET.end();
}
static inline void push_edge(int u,int v){
    if(u==v) return;
    if(u>v) std::swap(u,v);
    if(ESET.insert(ek(u,v)).second){
        E.push_back({u,v});
        if((int)DEG.size()!=N) DEG.assign(N,0);
        DEG[u]++; DEG[v]++;
        DEG_MAX = std::max(DEG_MAX, std::max(DEG[u], DEG[v]));
    }
}

// ============================================================================
// Parámetros FR
// ============================================================================
static inline float spring_k(){ return 2.0f / cbrtf((float)N); }

// ============================================================================
// Aleatorios en esfera
// ============================================================================
static Vec3 random_on_sphere(std::mt19937 &g){
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    while(true){
        float x = U(g), y = U(g), z = U(g);
        float r2 = x*x + y*y + z*z;
        if(r2 > 1e-6f && r2 <= 1.f){
            float r = sqrtf(r2);
            return {x/r, y/r, z/r};
        }
    }
}

// ============================================================================
// Forward: paso FR (lo usa el “reheat”)
// ============================================================================
static void fr_step();

// ============================================================================
// Recalentar y mini-relajar (para reacomodo visible tras cambios)
// ============================================================================
static void reheat_and_relax(float Tmin, float factor, int relax_iters){
    T = std::max(T, Tmin) * factor;
    bool oldCOOL = COOL;
    COOL = false;
    for(int i=0; i<relax_iters; ++i) fr_step();
    COOL = oldCOOL;
}


static void nudge_after_param_change(){
    // sube T bastante y déjalo respirar un poco
    T = std::max(T, 0.16f) * 1.65f;
    bool oldCOOL = COOL;
    COOL = false;
    for(int i=0; i<90; ++i) fr_step();   // antes 45, ahora 90
    COOL = oldCOOL;

    heat_cooldown_frames = 120;          // ~2s sin enfriar (antes 60)
    P_smooth = P;                         // evita ghosting del suavizado
}

static void set_force_model(ForceModel fm){
    bool into_SE = (FORCE_MODE != FM_SPRINGELEC && fm == FM_SPRINGELEC);
    FORCE_MODE = fm;
    if (into_SE) {
        SE_L = 0.9f * spring_k();  // solo al entrar
    }
    nudge_after_param_change();
}

// ============================================================================
// Estética por grado
// ============================================================================
static void recompute_degrees(){
    DEG.assign(N, 0);
    for(const Edge& e : E){
        if(e.u>=0 && e.u<N) DEG[e.u]++;
        if(e.v>=0 && e.v<N) DEG[e.v]++;
    }
    DEG_MAX = 1;
    for(int d : DEG) DEG_MAX = std::max(DEG_MAX, d);
}

static void degree_color(int d, float& r, float& g, float& b){
    float t = (float)d / (float)max(1, DEG_MAX);
    auto L=[](float a,float b,float u){return a+(b-a)*u;};
    if(t<0.5f){ float u=t/0.5f;  r=L(0.10f,0.10f,u); g=L(0.95f,0.90f,u); b=L(0.95f,0.25f,u); }
    else if(t<0.8f){ float u=(t-0.5f)/0.3f; r=L(0.10f,0.95f,u); g=L(0.90f,0.55f,u); b=L(0.25f,0.15f,u); }
    else { float u=(t-0.8f)/0.2f; r=L(0.95f,0.95f,u); g=L(0.55f,0.10f,u); b=L(0.15f,0.10f,u); }
}

static float node_radius_for(int i){
    const float base  = 0.025f;
    const float alpha = 0.010f;
    return base + alpha * ((float)DEG[i] / (float)std::max(1,DEG_MAX));
}

// ============================================================================
// k-means (con inicialización k-means++)
// ============================================================================
static void kmeans_assign(int K, int max_iters, bool use_smooth){
    if(K<=0) return;
    const auto& POS = (use_smooth && USE_SMOOTH) ? P_smooth : P;
    int n = (int)POS.size();
    if(n==0) return;

    COMM.assign(n, 0);
    CENT.clear(); CENT.reserve(K);

    // k-means++ init
    std::uniform_int_distribution<int> U0(0, n-1);
    int first = U0(RNG_KM);
    CENT.push_back(POS[first]);

    std::vector<float> mind(n, 1e30f);
    for(int c=1; c<K; ++c){
        for(int i=0;i<n;i++){
            float d = dist2(POS[i], CENT.back());
            if(d < mind[i]) mind[i] = d;
        }
        double sum=0; for(float v: mind) sum += v;
        if(sum <= 1e-9) { CENT.push_back(POS[U0(RNG_KM)]); continue; }

        std::uniform_real_distribution<double> U(0.0, sum);
        double r = U(RNG_KM), acc=0;
        int pick = n-1;
        for(int i=0;i<n;i++){ acc += mind[i]; if(acc>=r){ pick=i; break; } }
        CENT.push_back(POS[pick]);
    }

    // Lloyd
    for(int it=0; it<max_iters; ++it){
        bool changed=false;
        for(int i=0;i<n;i++){
            float best=1e30f; int bc=0;
            for(int c=0;c<K;c++){
                float d = dist2(POS[i], CENT[c]);
                if(d<best){ best=d; bc=c; }
            }
            if(COMM[i]!=bc){ COMM[i]=bc; changed=true; }
        }
        std::vector<Vec3> acc(K, {0,0,0});
        std::vector<int>  cnt(K, 0);
        for(int i=0;i<n;i++){
            int c=COMM[i];
            acc[c].x += POS[i].x; acc[c].y += POS[i].y; acc[c].z += POS[i].z;
            cnt[c]++;
        }
        for(int c=0;c<K;c++){
            if(cnt[c]>0){
                CENT[c].x = acc[c].x/cnt[c];
                CENT[c].y = acc[c].y/cnt[c];
                CENT[c].z = acc[c].z/cnt[c];
            }
        }
        if(!changed) break;
    }
    C_COMM = K;
}

// ============================================================================
// Reset del grafo
// ============================================================================
static void reset_graph(){
    std::mt19937 rng(12345678u); // reproducible

    P.assign(N, {0,0,0});
    D.assign(N, {0,0,0});
    E.clear();
    ESET.clear();

    // anillo + long range
    for(int i=0;i<N;i++) push_edge(i,(i+1)%N);
    for(int i=0;i<N/3;i++) push_edge(i,(i+N/2)%N);
    recompute_degrees();

    for(int i=0;i<N;i++) P[i] = random_on_sphere(rng);

    T = 0.12f;
    P_prev   = P;
    P_smooth = P;
    LAST_ADDED = {-1,-1};

    COMM.assign(N, 0);
    C_COMM = 0;
    SE_L = spring_k() * 0.9f;  // punto de partida razonable


    // k-means inicial “bonito”
    kmeans_assign(K_COMM, 15, /*use_smooth=*/true);

    printf("Reset: E=%zu  DEG_MAX=%d  N=%d\n", E.size(), DEG_MAX, N);
}

// ============================================================================
// Utilidad: re-centra para evitar drift
// ============================================================================
static void recenter(){
    Vec3 c{0,0,0};
    for(auto &p: P){ c.x += p.x; c.y += p.y; c.z += p.z; }
    c.x/=N; c.y/=N; c.z/=N;
    for(auto &p: P){ p.x -= c.x; p.y -= c.y; p.z -= c.z; }
}

// ============================================================================
// Δ medio
// ============================================================================
static float mean_disp(const vector<Vec3>& A, const vector<Vec3>& B){
    float acc=0.f;
    for(int i=0;i<N;i++){
        float dx=B[i].x-A[i].x, dy=B[i].y-A[i].y, dz=B[i].z-A[i].z;
        acc += sqrtf(dx*dx + dy*dy + dz*dz);
    }
    return acc/N;
}

// ============================================================================
// Aristas aleatorias (con reheat y k-means opcional)
// ============================================================================
static void add_bridge_edge_random(){
    std::mt19937 rng((unsigned)time(nullptr));
    std::uniform_int_distribution<int> U(0, N-1);
    for(int t=0;t<200;t++){
        int u = U(rng), v = U(rng);
        if(u==v) continue;
        if(!has_edge(u,v)){
            push_edge(u,v);
            LAST_ADDED = {u,v};
            if(T < T_MIN) reheat_and_relax(0.12f, 1.25f, 40);
            else          reheat_and_relax(0.12f, 1.35f, 40);
            // if (COLOR_MODE == BY_COMMUNITY) kmeans_assign(K_COMM, 10, true);
            return;
        }
    }
}

// ============================================================================
// cargar datos
// ============================================================================
// --- forward declarations necesarias para el loader ---
static void recompute_degrees();
static void kmeans_assign(int K, int max_iters=20, bool use_smooth=true);
static void reset_graph();
static bool load_edge_list_csv(const char* path);
static bool load_graph_from_file_or_default(const char* maybePath){
    if(maybePath && *maybePath){
        if(load_edge_list_csv(maybePath)) return true;
        printf("[loader] Falló cargar '%s', usando sintético.\n", maybePath);
    }
    reset_graph();
    return true;
}


// ===== Helpers de parsing robusto =====
static inline std::string trim_copy(std::string s){
    auto is_ws = [](unsigned char c){ return std::isspace(c) || c=='\r'; };
    while(!s.empty() && is_ws((unsigned char)s.front())) s.erase(s.begin());
    while(!s.empty() && is_ws((unsigned char)s.back()))  s.pop_back();
    return s;
}

static inline std::string strip_quotes(std::string s){
    s = trim_copy(s);
    if(!s.empty() && (s.front()=='"' || s.front()=='\'')) s.erase(s.begin());
    if(!s.empty() && (s.back()=='"'  || s.back()=='\'' )) s.pop_back();
    return s;
}

// Acepta: encabezados, comas/; tabs, comillas, y cualquier etiqueta.
// Devuelve los PRIMEROS dos tokens como strings limpios.
static bool parse_edge_line_generic(const std::string& raw, std::string& A, std::string& B){
    std::string s = trim_copy(raw);
    if(s.empty()) return false;
    if(s[0]=='#' || s[0]=='%') return false;   // comentario

    // Normaliza separadores
    for(char& c : s){
        if(c==',' || c==';' || c=='\t') c=' ';
    }
    std::istringstream iss(s);
    std::string a,b;
    if(!(iss >> a)) return false;          // línea vacía
    if(!(iss >> b)) return false;          // header tipo "source target"
    a = strip_quotes(a);
    b = strip_quotes(b);

    // descarta header típico
    std::string al = a; for(char& c: al) c=std::tolower((unsigned char)c);
    std::string bl = b; for(char& c: bl) c=std::tolower((unsigned char)c);
    if( (al=="source" && bl=="target") || (al=="src" && bl=="dst") ) return false;

    if(a.empty() || b.empty()) return false;
    return true;
}


static bool load_edge_list_csv(const char* path){
    std::ifstream fin(path);
    if(!fin.is_open()){
        printf("[loader] No pude abrir '%s'\n", path);
        return false;
    }
    printf("[loader] Leyendo edges desde '%s'...\n", path);

    std::vector<std::pair<int,int>> edges;
    edges.reserve(10000);

    std::unordered_map<std::string,int> idmap;
    idmap.reserve(4096);
    auto get_id = [&](const std::string& key)->int{
        auto it = idmap.find(key);
        if(it!=idmap.end()) return it->second;
        int id = (int)idmap.size();
        idmap.emplace(key, id);
        return id;
    };

    std::string line, a, b;
    int skipped = 0, parsed = 0;
    int max_id = -1; // <<<<<< NECESARIO para el fast path numérico

    while(std::getline(fin, line)){
        // ----- Fast path numérico: "u,v" (con espacios opcionales) -----
        {
            int uu, vv;
            if (std::sscanf(line.c_str(), " %d , %d ", &uu, &vv) == 2) {
                if (uu != vv) {
                    edges.emplace_back(uu, vv);
                    max_id = std::max(max_id, std::max(uu, vv));
                    ++parsed;
                } else {
                    ++skipped;
                }
                continue; // ya procesamos esta línea
            }
        }

        // ----- Parser genérico: "A,B" con etiquetas/strings -----
        if (!parse_edge_line_generic(line, a, b)) {
            ++skipped;
            continue;
        }
        int u = get_id(a);
        int v = get_id(b);
        if (u == v) { ++skipped; continue; }
        edges.emplace_back(u, v);
        ++parsed;
    }
    fin.close();

    if(parsed==0){
        printf("[loader] No encontré edges válidos. Skipped=%d\n", skipped);
        return false;
    }

    // Si vinieron enteros puros, max_id sirve como N-1; si vinieron etiquetas,
    // usamos idmap.size(). Toma el mayor de ambos por si mezclan.
    int inferredN = std::max(max_id + 1, (int)idmap.size());
    N = std::max(1, inferredN);

    // Estructuras
    P.assign(N, {0,0,0});
    D.assign(N, {0,0,0});
    P_prev.assign(N, {0,0,0});
    P_smooth.assign(N, {0,0,0});
    DEG.assign(N, 0);
    E.clear(); ESET.clear();
    LAST_ADDED = {-1,-1};

    for(auto &e : edges) push_edge(e.first, e.second);
    recompute_degrees();

    // posiciones iniciales
    std::mt19937 rng(12345678u);
    for(int i=0;i<N;i++) P[i] = random_on_sphere(rng);
    P_prev = P; P_smooth = P;

    T = 0.12f;
    SE_L = spring_k() * 0.9f;
    COMM.assign(N, 0);
    C_COMM = 0;
    kmeans_assign(K_COMM, 15, /*use_smooth=*/true);

    printf("[loader] Cargado: N=%d  |E|=%zu  skipped=%d (labels únicos=%zu, max_id=%d)\n",
           N, E.size(), skipped, idmap.size(), max_id);
    return true;
}




// ============================================================================
// Paso FR principal
// ============================================================================
static void fr_step(){
    const float k = spring_k();
    for(int i=0;i<N;i++) D[i] = {0,0,0};

    // --- REPELUSIÓN entre todos los pares (i<j)
    for(int i=0;i<N;i++){
        for(int j=i+1;j<N;j++){
            Vec3 delta = sub(P[i], P[j]);
            float d = len(delta);
            Vec3 dir = norm(delta);

            float f = 0.0f;
            switch (FORCE_MODE){
                case FM_FR:
                    // FR clásico: k^2 / d
                    f = (k*k) / d;
                    break;
                case FM_LINLOG:
                    // LinLog: 1/d también (mantén escala con k^2)
                    f = (k*k) / d;
                    break;
                case FM_SPRINGELEC:
                    // Coulomb 1/d^2 con constante
                    f = SE_Cr / (d*d);
                    break;
            }

            Vec3 F = mul(dir, f);
            D[i] = add(D[i], F);
            D[j] = sub(D[j], F);
        }
    }

    // --- ATRACCIÓN solo en aristas
    for(const Edge& e: E){
        Vec3 delta = sub(P[e.u], P[e.v]);
        float d  = len(delta);
        Vec3 dir = norm(delta);

        float f = 0.0f;
        switch (FORCE_MODE){
            case FM_FR:
                // FR clásico: d^2 / k
                f = (d*d) / k;
                break;
            case FM_LINLOG:
                // LinLog: (1/k) * log(1 + d)
                f = logf(1.0f + d) / k;
                break;
            case FM_SPRINGELEC:
                // muelle lineal: Ca * (d - L)
                f = SE_Ca * (d - SE_L);
                break;
        }

        // OJO: en atracción restamos en u y sumamos en v
        Vec3 F = mul(dir, f);
        D[e.u] = sub(D[e.u], F);
        D[e.v] = add(D[e.v], F);
    }

    // --- Integración
    for (int i=0;i<N;i++){
        float L = len(D[i]);
        Vec3 step = (L>1e-9f) ? mul(D[i], min(T/L, 1.0f)) : Vec3{0,0,0};
        P[i] = add(P[i], step);

        // caja más amplia en SE
        float clampVal = (FORCE_MODE==FM_SPRINGELEC) ? 2.2f : 1.4f;
        P[i].x = clampf(P[i].x, -clampVal, clampVal);
        P[i].y = clampf(P[i].y, -clampVal, clampVal);
        P[i].z = clampf(P[i].z, -clampVal, clampVal);
    }

    // resorte radial suave (más suave en SE)
    const float r0 = 1.0f;
    const float kr = (FORCE_MODE==FM_SPRINGELEC) ? 0.003f : 0.01f;
    for(int i=0;i<N;i++){
        float r = len(P[i]);
        if(r > 1e-6f){
            float extra = kr * (r0 - r);
            P[i].x += P[i].x * extra;
            P[i].y += P[i].y * extra;
            P[i].z += P[i].z * extra;
        }
    }
    recenter();

}

// ============================================================================
// Proyección y cámara (cámara fija, rotamos el modelo)
// ============================================================================
static void set_perspective(int w,int h){
    glViewport(0,0,w,h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float aspect = (h>0) ? (float)w/(float)h : 1.0f;
    gluPerspective(45.0, aspect, 0.1, 100.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

static void apply_camera(){
    glLoadIdentity();
    // cámara fija viendo al origen desde +Z
    gluLookAt(0, 0, cam_radius,  0,0,0,  0,1,0);
}

// ============================================================================
// Ejes 3D
// ============================================================================
static void draw_axes3D(float L=1.0f){
    glLineWidth(2.0f);
    glBegin(GL_LINES);
      glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(L,0,0);
      glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,L,0);
      glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,L);
    glEnd();
}

// === Helpers HUD ===
static void hudBegin2D(int w, int h){
    glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT);
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, w, 0, h);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
}

static void hudEnd2D(){
    glPopMatrix();                 // modelview
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();                 // projection
    glMatrixMode(GL_MODELVIEW);
    glPopAttrib();
}

static void hudPrint(int x, int y, const char* s){
    glRasterPos2i(x, y);
    for(const char* p=s; *p; ++p) glutBitmapCharacter(GLUT_BITMAP_8_BY_13, *p);
}

static int hudTextHeight(){ return 14; } // aprox para GLUT_BITMAP_8_BY_13

static void hudBox(int x, int y, int w, int h, const float bg[4]){
    glColor4fv(bg);
    glBegin(GL_QUADS);
      glVertex2i(x,   y);
      glVertex2i(x+w, y);
      glVertex2i(x+w, y+h);
      glVertex2i(x,   y+h);
    glEnd();
}


// ============================================================================
// Posición para dibujar (suavizada si está activo)
// ============================================================================
static inline const Vec3& Q(int i){
    return USE_SMOOTH ? P_smooth[i] : P[i];
}

// ============================================================================
// Luces
// ============================================================================
static void setup_lighting(){
    if (USE_LIGHT){
        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        glEnable(GL_COLOR_MATERIAL);
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

        const GLfloat pos[4]  = { 1.0f, 1.0f, 2.0f, 0.0f }; // direccional
        const GLfloat amb[4]  = { 0.15f, 0.15f, 0.15f, 1.0f };
        const GLfloat diff[4] = { 0.85f, 0.85f, 0.85f, 1.0f };
        const GLfloat spec[4] = { 0.20f, 0.20f, 0.20f, 1.0f };
        glLightfv(GL_LIGHT0, GL_POSITION, pos);
        glLightfv(GL_LIGHT0, GL_AMBIENT,  amb);
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  diff);
        glLightfv(GL_LIGHT0, GL_SPECULAR, spec);

        glEnable(GL_NORMALIZE);
        glShadeModel(GL_SMOOTH);
        GLfloat shin[1] = { 32.0f };
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shin);
    } else {
        glDisable(GL_LIGHT0);
        glDisable(GL_COLOR_MATERIAL);
        glDisable(GL_LIGHTING);
    }
}

// ============================================================================
// Dibujo del grafo (rotando el MODELO)
// ============================================================================
static void degree_or_comm_color(int i, float& r,float& g,float& b){
    if (COLOR_MODE == BY_COMMUNITY && C_COMM > 0 && i < (int)COMM.size())
        community_color(COMM[i], r,g,b);
    else
        degree_color(DEG[i], r,g,b);
}

static void draw_graph3D(){
    // aplicar rotaciones del modelo
    glPushMatrix();
    glRotatef(model_pitch, 1,0,0);
    glRotatef(model_yaw,   0,1,0);

    // aristas
    glDisable(GL_LIGHTING);
    glColor3f(0.35f,0.35f,0.35f);
    glLineWidth(1.2f);
    glBegin(GL_LINES);
    for(const Edge& e: E){
        glVertex3f(Q(e.u).x, Q(e.u).y, Q(e.u).z);
        glVertex3f(Q(e.v).x, Q(e.v).y, Q(e.v).z);
    }
    glEnd();

    // resalte última arista
    if(LAST_ADDED.u>=0){
        glDisable(GL_LIGHTING);
        glLineWidth(3.0f);
        glColor3f(1.0f, 0.2f, 0.2f);
        glBegin(GL_LINES);
        glVertex3f(Q(LAST_ADDED.u).x, Q(LAST_ADDED.u).y, Q(LAST_ADDED.u).z);
        glVertex3f(Q(LAST_ADDED.v).x, Q(LAST_ADDED.v).y, Q(LAST_ADDED.v).z);
        glEnd();
        glLineWidth(1.2f);
    }

    // nodos
    if (USE_LIGHT){
        glEnable(GL_LIGHTING);
        glEnable(GL_COLOR_MATERIAL);
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    }else{
        glDisable(GL_LIGHTING);
        glDisable(GL_COLOR_MATERIAL);
    }

    for (int i = 0; i < N; ++i) {
        float r,g,b; degree_or_comm_color(i, r,g,b);
        glColor3f(r,g,b);
        glPushMatrix();
        glTranslatef(Q(i).x, Q(i).y, Q(i).z);
        float rnode = 0.75f * node_radius_for(i);
        glutSolidSphere(rnode, node_slices, node_stacks);
        glPopMatrix();
    }

    glDisable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);
    glPopMatrix(); // deshace rotaciones del modelo
}

// ============================================================================
// HUD minimalista
// ============================================================================
static void drawHUD(int winW, int winH) {
    hudBegin2D(winW, winH);

    const int lineH = hudTextHeight();
    const int pad   = HUD_PAD;

    // ==== Panel Métricas (arriba-izquierda) ====
    int x = 8, y = winH - 10;
    char line[256];

    // 1a línea
    const char* forceName = (FORCE_MODE==FM_FR) ? "FR" :
                            (FORCE_MODE==FM_LINLOG) ? "LinLog" : "SpringElec";
    snprintf(line, sizeof(line),
       "N:%d  |E|:%zu  DEG_MAX:%d  FPS:%.1f  Δ:%.5f",
       N, E.size(), DEG_MAX, g_fps, g_lastDelta);
    int w1 = 8 + (int)strlen(line)*8 + 2*pad;

    // 2a línea
    char line2[256];
    snprintf(line2, sizeof(line2),
       "T:%.3f  cool:%s  it/frame:%d  cam(r=%.2f yaw=%.1f pitch=%.1f)",
       T, COOL?"on":"off", iters_per_frame, cam_radius, model_yaw, model_pitch);
    int w2 = 8 + (int)strlen(line2)*8 + 2*pad;

    // 3a línea
    char line3[256];
    snprintf(line3, sizeof(line3),
       "Color:%s(K=%d)  smooth:%s(α=%.2f)  light:%s  force:%s",
       (COLOR_MODE==BY_COMMUNITY?"kmeans":"degree"), K_COMM,
       (USE_SMOOTH?"on":"off"), ALPHA,
       (USE_LIGHT?"on":"off"), forceName);
    int w3 = 8 + (int)strlen(line3)*8 + 2*pad;

    // 4a línea (solo SE)
    char line4[256] = "";
    int w4 = 0, lines = 3;
    if (FORCE_MODE == FM_SPRINGELEC){
        snprintf(line4, sizeof(line4), "[SE] Cr=%.3f  Ca=%.3f  L=%.3f", SE_Cr, SE_Ca, SE_L);
        w4 = 8 + (int)strlen(line4)*8 + 2*pad;
        lines = 4;
    }

    int boxW = std::max(std::max(w1,w2), std::max(w3,w4));
    int boxH = lines*lineH + 2*pad;

    hudBox(4, winH - boxH - 4, boxW, boxH, HUD_BG);
    glColor3fv(HUD_FG);

    int ty = winH - pad - 6;
    hudPrint(x+pad, ty, line);      ty -= lineH;
    hudPrint(x+pad, ty, line2);     ty -= lineH;
    hudPrint(x+pad, ty, line3);     ty -= lineH;
    if (FORCE_MODE == FM_SPRINGELEC) hudPrint(x+pad, ty, line4);

    // ==== Panel Ayuda/Controles (abajo-izquierda) ====
    if (HUD_HELP){
        const char* help[] = {
            "[Controles]",
            "Espacio: pausa | r: reset | +/-: iters | t/T: temperatura | g: cooling",
            "b/B: puentes | c: color degree/kmeans | k/p/P: k-means | s: smooth , .: alpha",
            "q/Q: calidad nodos | [ ]: zoom | Flechas: rotar | o: auto-rot | x: screenshot",
            "m/1/2/3: modelo de fuerzas (FR/LinLog/SpringElec)",
            "SpringElec: u/U repulsion  y/Y rigidez  h/H longitud L",
            "? : ocultar/mostrar esta ayuda"
        };
        int n = (int)(sizeof(help)/sizeof(help[0]));
        int maxw = 0;
        for(int i=0;i<n;i++){
            int wi = (int)strlen(help[i])*8 + 2*pad;
            maxw = std::max(maxw, wi);
        }
        int hx = 4, hy = 4, hh = n*lineH + 2*pad;
        hudBox(hx, hy, maxw, hh, HUD_BG);
        glColor3fv(HUD_FG);
        int ty2 = hy + hh - pad - 10;
        for(int i=0;i<n;i++){
            hudPrint(hx+pad, ty2, help[i]);
            ty2 -= lineH;
        }
    } else {
        // mini-hint para recuperar ayuda
        const char* hint = "[?] ayuda";
        int ww = (int)strlen(hint)*8 + 2*pad;
        int hh = lineH + 2*pad;
        hudBox(4, 4, ww, hh, HUD_BG);
        glColor3fv(HUD_FG);
        hudPrint(4+pad, 4+pad, hint);
    }

    hudEnd2D();
}


// ============================================================================
// Captura de pantalla (PPM sencillo)
// ============================================================================
static void save_ppm_backbuffer(const char* path){
    int w = glutGet(GLUT_WINDOW_WIDTH);
    int h = glutGet(GLUT_WINDOW_HEIGHT);
    std::vector<unsigned char> pix(3*w*h);

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_BACK);                   // <- clave
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, pix.data());

    FILE* f = fopen(path, "wb");
    if(!f) return;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    // PPM espera origen arriba; OpenGL entrega abajo-izq -> flip vertical
    for(int y = h-1; y >= 0; --y){
        fwrite(&pix[y*3*w], 1, 3*w, f);
    }
    fclose(f);
}


// ============================================================================
// Display
// ============================================================================
static void display(){
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // 3D primero
    apply_camera();
    setup_lighting();
    draw_axes3D();
    draw_graph3D();

    // HUD después
    int w = glutGet(GLUT_WINDOW_WIDTH);
    int h = glutGet(GLUT_WINDOW_HEIGHT);
    drawHUD(w, h);


    if (g_want_snap) {
        save_ppm_backbuffer("snap.ppm");
        g_want_snap = false;
    }

    glutSwapBuffers();
}

// ============================================================================
// Idle (física + suavizado + métricas + auto-kmeans por calma)
// ============================================================================
static void idle(){
    if(!RUN){ glutPostRedisplay(); return; }

    if((int)P_prev.size()!=N) P_prev = P;

    for(int i=0;i<iters_per_frame;i++) fr_step();

    // suavizado
    for (int i = 0; i < N; ++i) {
        P_smooth[i].x = (1.0f - ALPHA) * P_smooth[i].x + ALPHA * P[i].x;
        P_smooth[i].y = (1.0f - ALPHA) * P_smooth[i].y + ALPHA * P[i].y;
        P_smooth[i].z = (1.0f - ALPHA) * P_smooth[i].z + ALPHA * P[i].z;
    }

    if (heat_cooldown_frames > 0) {
        --heat_cooldown_frames;          // todavía no enfríes
    } else if (COOL && T > T_MIN) {
        T *= 0.985f;
    }


    float d = mean_disp(P_prev, P);
    P_prev = P;
    g_lastDelta = d;

    // FPS (ventana 500ms)
    int now = glutGet(GLUT_ELAPSED_TIME);
    ++g_frames;
    if (now - g_prevTime >= 500) {
        g_fps = g_frames * 1000.0f / float(now - g_prevTime);
        g_frames = 0;
        g_prevTime = now;
    }

    // auto-rotación del MODELO (no de la cámara)
    if (AUTO_ROT) {
        model_yaw += 0.15f;
        if (model_yaw > 360.0f) model_yaw -= 360.0f;
    }

    // auto-kmeans cuando se calma
    static int calmFrames=0;
    static int coolDown=0;
    if (COLOR_MODE==BY_COMMUNITY){
        if (g_lastDelta < 0.003f) calmFrames++; else calmFrames=0;
        if (coolDown>0) coolDown--;
        // if (calmFrames>=30 && coolDown==0){
        //     kmeans_assign(K_COMM, 10, /*use_smooth=*/true);
        //     calmFrames = 0;
        //     coolDown   = 120; // ~2s a 60 FPS
        // }
    } else {
        calmFrames=0; coolDown=0;
    }


    glutPostRedisplay();
}

// ============================================================================
// Teclado (caracteres)
// ============================================================================
static void keyboard(unsigned char key,int,int){
    if(key==' '){ RUN=!RUN; }

    else if (key=='t') { if (T < T_MIN) T = T_KICK; else T = std::min(T*1.10f + 0.002f, T_MAX); }
    else if (key=='T') { T = std::max(T*0.90f, T_MIN); }

    else if (key=='c') { COLOR_MODE = (COLOR_MODE==BY_DEGREE) ? BY_COMMUNITY : BY_DEGREE; }

    else if (key=='k') { kmeans_assign(K_COMM, 20, /*use_smooth=*/true); }
    else if (key=='p') { K_COMM = std::min(K_COMM+1, 12); kmeans_assign(K_COMM, 20, true); }
    else if (key=='P') { K_COMM = std::max(K_COMM-1, 2);  kmeans_assign(K_COMM, 20, true); }

    else if (key=='r' || key=='R'){ reset_graph(); }

    else if (key=='+'){ iters_per_frame = min(64, iters_per_frame+1); }
    else if (key=='-'){ iters_per_frame = max(1, iters_per_frame-1); }

    else if (key=='g'){ COOL=!COOL; }
    else if (key=='b'){ add_bridge_edge_random(); }
    else if (key=='B'){ for(int k=0;k<5;++k) add_bridge_edge_random(); reheat_and_relax(0.12f,1.6f,120); }

    else if (key=='v' || key=='V'){ USE_LIGHT = !USE_LIGHT; glutPostRedisplay(); }
    else if (key=='s' || key=='S'){ USE_SMOOTH = !USE_SMOOTH; }

    else if (key==','){ ALPHA = std::max(0.0f, ALPHA - 0.05f); }
    else if (key=='.'){ ALPHA = std::min(1.0f, ALPHA + 0.05f); }

    else if (key=='['){ cam_radius = max(0.5f, cam_radius*0.9f); }   // zoom in
    else if (key==']'){ cam_radius = min(50.0f, cam_radius*1.1f); }  // zoom out

    else if (key=='o'){ AUTO_ROT = !AUTO_ROT; }

    else if (key=='q'){ node_slices=8;  node_stacks=8;  glLineWidth(1.0f); }
    else if (key=='Q'){ node_slices=14; node_stacks=14; glLineWidth(1.5f); }

    else if (key=='x' || key=='X') { g_want_snap = true; }
    else if (key=='?'){ HUD_HELP = !HUD_HELP; }

    else if (key=='m' || key=='M') {
        set_force_model( (ForceModel)((FORCE_MODE + 1) % 3) ); // FR -> LinLog -> SE -> ...
    }
    else if (key=='1') { set_force_model(FM_FR);         }
    else if (key=='2') { set_force_model(FM_LINLOG);     }
    else if (key=='3') { set_force_model(FM_SPRINGELEC); }



    else if (FORCE_MODE == FM_SPRINGELEC &&
            (key=='u' || key=='U' || key=='y' || key=='Y' || key=='h' || key=='H')) {

        const float k = spring_k();

        if (key=='u') {
            SE_Cr *= 1.50f;          // antes 1.15f
            SE_L  *= 1.05f;          // al subir repulsión, estira L un 5%
        }
        if (key=='U') {
            SE_Cr /= 1.50f;
            SE_L  /= 1.05f;
        }

        if (key=='y') SE_Ca *= 1.25f;   // un pelín más notorio
        if (key=='Y') SE_Ca /= 1.25f;

        if (key=='h') SE_L  *= 1.15f;   // que sí se vea
        if (key=='H') SE_L  /= 1.15f;

        // límites sanos
        SE_Cr = clampf(SE_Cr, 1e-4f, 200.0f);
        SE_Ca = clampf(SE_Ca, 1e-4f, 200.0f);
        SE_L  = clampf(SE_L,  0.10f*k, 5.0f*k);

        nudge_after_param_change();
        printf("[SE] Cr=%.4f  Ca=%.4f  L=%.4f  (k=%.4f)\n", SE_Cr, SE_Ca, SE_L, k);
    }



    // Preset de demo reproducible
    else if (key=='F'){
        reset_graph();
        COLOR_MODE = BY_COMMUNITY;
        kmeans_assign(K_COMM, 20, true);
        add_bridge_edge_random();
        // pequeño espectáculo
        for(int i=0;i<2;i++) add_bridge_edge_random();
        reheat_and_relax(0.12f, 1.35f, 80);
    }

    else if(key==27){ exit(0); } // ESC

    printf("T=%.3f  it/frame=%d  cooling=%d  edges=%zu  cam(r=%.2f)  alpha=%.2f  mode=%s(K=%d)\n",
           T, iters_per_frame, COOL?1:0, E.size(), cam_radius, ALPHA,
           (COLOR_MODE==BY_COMMUNITY?"kmeans":"degree"), K_COMM);
}

// ============================================================================
// Teclas especiales (flechas) para rotar el MODELO
// ============================================================================
static void special(int key, int, int){
    const float yaw_step   = 3.0f;
    const float pitch_step = 3.0f;
    if (key == GLUT_KEY_LEFT)  model_yaw   -= yaw_step;
    if (key == GLUT_KEY_RIGHT) model_yaw   += yaw_step;
    if (key == GLUT_KEY_UP)    model_pitch = clampf(model_pitch + pitch_step, -89.0f, 89.0f);
    if (key == GLUT_KEY_DOWN)  model_pitch = clampf(model_pitch - pitch_step, -89.0f, 89.0f);
    glutPostRedisplay();
}

// ============================================================================
// Reshape
// ============================================================================
static void reshape(int w,int h){
    glEnable(GL_DEPTH_TEST);
    set_perspective(w,h);
    g_win_w = w;
    g_win_h = h;
}

// ============================================================================
// main
// ============================================================================
int main(int argc,char**argv){
    srand((unsigned)time(nullptr));
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(g_win_w, g_win_h);
    glutCreateWindow("COMP6838 — FR en 3D (layout + render)");
    glClearColor(1,1,1,1);

    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_POINT_SMOOTH);
    glDisable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);

    // reset_graph(); // lo reemplazamos por el loader
    // const char* dataset = "karate.csv";  // cambia la ruta si lo mueves
    // bool ok = load_graph_from_file_or_default(dataset);
    const char* dataset = "../data/karate_clean.csv";
    printf("[main] dataset='%s'\n", dataset);
    bool ok = load_graph_from_file_or_default(dataset);
    if(!ok){
        printf("[info] Usando grafo sintético por defecto.\n");
    }


    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(special);      // flechas para rotar el MODELO
    glutReshapeFunc(reshape);

    puts("Controles:"
         "  espacio pausa | r reset | +/- iters | t/T temp | g cooling |"
         "  b/B puentes | c color mode | k/p/P k-means | s suavizado | , . alpha |"
         "  q/Q calidad | [ ] zoom | flechas rotan modelo | o auto-rot | x screenshot | F demo | ESC salir");

    glutMainLoop();
    return 0;
}
