#ifndef LOCAL
#define NDEBUG
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <climits>
#include <cfloat>
#include <ctime>
#include <cassert>
#include <map>
#include <utility>
#include <set>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <sstream>
#include <complex>
#include <stack>
#include <queue>
#include <numeric>
#include <list>
#include <iomanip>
#include <fstream>
#include <bitset>

using namespace std;

#define foreach(it, c) for (__typeof__((c).begin()) it=(c).begin(); it != (c).end(); ++it)
template <typename T> void print_container(ostream& os, const T& c) { const char* _s = " "; if (!c.empty()) { __typeof__(c.begin()) last = --c.end(); foreach (it, c) { os << *it; if (it != last) os << _s; } } }
template <typename T> ostream& operator<<(ostream& os, const vector<T>& c) { print_container(os, c); return os; }
template <typename T> ostream& operator<<(ostream& os, const set<T>& c) { print_container(os, c); return os; }
template <typename T> ostream& operator<<(ostream& os, const multiset<T>& c) { print_container(os, c); return os; }
template <typename T> ostream& operator<<(ostream& os, const deque<T>& c) { print_container(os, c); return os; }
template <typename T, typename U> ostream& operator<<(ostream& os, const map<T, U>& c) { print_container(os, c); return os; }
template <typename T, typename U> ostream& operator<<(ostream& os, const pair<T, U>& p) { os << "(" << p.first << ", " << p.second << ")"; return os; }

template <typename T> void print(T a, int n, const string& split = " ") { for (int i = 0; i < n; i++) { cerr << a[i]; if (i + 1 != n) cerr << split; } cerr << endl; }
template <typename T> void print2d(T a, int w, int h, int width = -1, int br = 0) { for (int i = 0; i < h; ++i) { for (int j = 0; j < w; ++j) { if (width != -1) cerr.width(width); cerr << a[i][j] << ' '; } cerr << endl; } while (br--) cerr << endl; }
template <typename T> void input(T& a, int n) { for (int i = 0; i < n; ++i) cin >> a[i]; }
#define dump(v) (cerr << #v << ": " << v << endl)
// #define dump(v)

#define rep(i, n) for (int i = 0; i < (int)(n); ++i)
#define erep(i, n) for (int i = 0; i <= (int)(n); ++i)
#define all(a) (a).begin(), (a).end()
#define rall(a) (a).rbegin(), (a).rend()
#define clr(a, x) memset(a, x, sizeof(a))
#define sz(a) ((int)(a).size())
#define mp(a, b) make_pair(a, b)
#define ten(n) ((long long)(1e##n))

template <typename T, typename U> void upmin(T& a, const U& b) { a = min<T>(a, b); }
template <typename T, typename U> void upmax(T& a, const U& b) { a = max<T>(a, b); }
template <typename T> void uniq(T& a) { sort(a.begin(), a.end()); a.erase(unique(a.begin(), a.end()), a.end()); }
template <class T> string to_s(const T& a) { ostringstream os; os << a; return os.str(); }
template <class T> T to_T(const string& s) { istringstream is(s); T res; is >> res; return res; }
bool in_rect(int x, int y, int w, int h) { return 0 <= x && x < w && 0 <= y && y < h; }

typedef long long ll;
typedef pair<int, int> pint;
typedef unsigned long long ull;

// const int DX[] = { 0, 1, 0, -1 };
// const int DY[] = { 1, 0, -1, 0 };

const int DX[] = { -1, +0, +1, +0 };
const int DY[] = { +0, +1, +0, -1 };
const char* S_DIR = "LDRU";




ull rdtsc()
{
#ifdef __amd64
    ull a, d;
    __asm__ volatile ("rdtsc" : "=a" (a), "=d" (d));
    return (d<<32) | a;
#else
    ull x;
    __asm__ volatile ("rdtsc" : "=A" (x));
    return x;
#endif
}
#ifdef LOCAL
const double CYCLES_PER_SEC = 3.30198e9;
#else
const double CYCLES_PER_SEC = 2.5e9;
#endif
double get_absolute_sec()
{
    return (double)rdtsc() / CYCLES_PER_SEC;
}
#ifdef _MSC_VER
#include <Windows.h>
    double get_ms() { return (double)GetTickCount64() / 1000; }
#else
#include <sys/time.h>
    double get_ms() { struct timeval t; gettimeofday(&t, NULL); return (double)t.tv_sec * 1000 + (double)t.tv_usec / 1000; }
#endif

#ifndef LOCAL
#define USE_RDTSC
#endif
class Timer
{
private:
    double start_time;
    double elapsed;

#ifdef USE_RDTSC
    double get_sec() { return get_absolute_sec(); }
#else
    double get_sec() { return get_ms() / 1000; }
#endif

public:
    Timer() {}

    void start() { start_time = get_sec(); }
    double get_elapsed() { return elapsed = get_sec() - start_time; }
};
Timer g_timer;
#ifdef LOCAL
#define USE_TIMER
#ifdef USE_TIMER
const double G_TL_SEC = 9.5;
#else
const double G_TL_SEC = 1e9;
#endif
#else
#define USE_TIMER
const double G_TL_SEC = 9.8;
#endif


struct Pos
{
    int x, y;
    Pos(int x, int y)
        : x(x), y(y)
    {
    }
    Pos()
        : x(0), y(0)
    {
    }

    bool operator==(const Pos& other) const
    {
        return x == other.x && y == other.y;
    }
    bool operator !=(const Pos& other) const
    {
        return x != other.x || y != other.y;
    }

    void operator+=(const Pos& other)
    {
        x += other.x;
        y += other.y;
    }
    void operator-=(const Pos& other)
    {
        x -= other.x;
        y -= other.y;
    }

    Pos operator+(const Pos& other) const
    {
        Pos res = *this;
        res += other;
        return res;
    }
    Pos operator-(const Pos& other) const
    {
        Pos res = *this;
        res -= other;
        return res;
    }
    Pos operator*(int a) const
    {
        return Pos(x * a, y * a);
    }

    bool operator<(const Pos& other) const
    {
        if (x != other.x)
            return x < other.x;
        else
            return y < other.y;
    }

    int dist(const Pos& p) const
    {
        return abs(p.x - x) + abs(p.y - y);
    }

    Pos next(int dir) const
    {
        return Pos(x + DX[dir], y + DY[dir]);
    }

    void move(int dir)
    {
        assert(0 <= dir && dir < 4);
        x += DX[dir];
        y += DY[dir];
    }

    int dir(const Pos& to) const
    {
        rep(dir, 4)
        {
            if (next(dir) == to)
                return dir;
        }
        assert(false);
        return -1;
    }

    uint pack() const
    {
        return (y << 7) | x;
    }
};
Pos operator*(int a, const Pos& pos)
{
    return pos * a;
}
ostream& operator<<(ostream& os, const Pos& pos)
{
    os << "(" << pos.x << ", " << pos.y << ")";
    return os;
}

int moved_dir(const Pos& from, const Pos& to)
{
    Pos diff = to - from;
    rep(dir, 4)
    {
        if (diff.x * DX[dir] + diff.y * DY[dir] > 0)
            return dir;
    }
    assert(false);
}

class Random
{
private:
    unsigned int  x, y, z, w;
public:
    Random(unsigned int x
             , unsigned int y
             , unsigned int z
             , unsigned int w)
        : x(x), y(y), z(z), w(w) { }
    Random()
        : x(123456789), y(362436069), z(521288629), w(88675123) { }
    Random(unsigned int seed)
        : x(123456789), y(362436069), z(521288629), w(seed) { }

    unsigned int next()
    {
        unsigned int t = x ^ (x << 11);
        x = y;
        y = z;
        z = w;
        return w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
    }

    int next_int() { return next(); }

    // [0, upper)
    int next_int(int upper) { return next() % upper; }

    // [low, high)
    int next_int(int low, int high) { return next_int(high - low) + low; }

    double next_double(double upper) { return upper * next() / UINT_MAX; }
    double next_double(double low, double high) { return next_double(high - low) + low; }

    template <typename T>
    int select(const vector<T>& ratio)
    {
        T sum = accumulate(ratio.begin(), ratio.end(), (T)0);
        double v = next_double(sum) + 1e-6;
        for (int i = 0; i < (int)ratio.size(); ++i)
        {
            v -= ratio[i];
            if (v <= 0)
                return i;
        }
        return (int)ratio.size() - 1;
    }
};
Random g_rand;

const int WALL = 10;
const int EMPTY = 11;
class Board
{
public:
    Board(){}
    Board(const vector<string>& s) :
        h(s.size()), w(s[0].size())
    {
        rep(y, h) rep(x, w)
        {
            if (s[y][x] == '#')
                a[y][x] = WALL;
            else if (isdigit(s[y][x]))
                a[y][x] = s[y][x] - '0';
            else
                a[y][x] = EMPTY;
        }
    }

    bool valid_access(int x, int y) const
    {
        return in_rect(x + 1, y + 1, w + 2, h + 2);
    }

    int height() const { return h; }
    int width() const { return w; }

    bool is_wall(int x, int y) const
    {
        assert(valid_access(x, y));
        return !in_rect(x, y, w, h) || a[y][x] == WALL;
    }
    bool is_wall(const Pos& p) const
    {
        return is_wall(p.x, p.y);
    }

    bool is_color(int x, int y) const
    {
        assert(valid_access(x, y));
        return in_rect(x, y, w, h) && a[y][x] < WALL;
    }
    bool is_color(const Pos& p) const
    {
        return is_color(p.x, p.y);
    }


    int color(int x, int y) const
    {
        assert(valid_access(x, y));
        assert(in_rect(x, y, w, h));
        return a[y][x];
    } 
    int color(const Pos& p) const
    {
        return color(p.x, p.y);
    }

    bool empty(int x, int y) const
    {
        assert(valid_access(x, y));
        return in_rect(x, y, w, h) && a[y][x] == EMPTY;
    }
    bool empty(const Pos& p) const
    {
        return empty(p.x, p.y);
    }

    bool is_obs(int x, int y) const
    {
        assert(valid_access(x, y));
        return !empty(x, y);
    }
    bool is_obs(const Pos& p) const
    {
        return is_obs(p.x, p.y);
    }

    void set(int x, int y, int c)
    {
        assert(in_rect(x, y, w, h));
        a[y][x] = c;
    }
    void set(const Pos& p, int c)
    {
        set(p.x, p.y, c);
    }

    void remove(int x, int y)
    {
        assert(is_color(x, y));
        a[y][x] = EMPTY;
    }
    void remove(const Pos& p)
    {
        remove(p.x, p.y);
    }

    void move(const Pos& from, const Pos& to)
    {
        assert(is_color(from));
        assert(!is_wall(to) && !is_color(to));

        int c = color(from);
        remove(from);
        set(to, c);
    }

    double game_score(const Board& target_board) const
    {
        int s = 0;
        int balls = 0;
        rep(y, h) rep(x, w)
        {
            if (target_board.is_color(x, y))
            {
                ++balls;
                if (is_color(x, y))
                {
                    if (target_board.color(x, y) == color(x, y))
                        s += 2;
                    else
                        s += 1;
                }
            }
        }
        return s / 2.0 / balls;
    }

private:
    int h, w;
    int a[64][64];
};
Board target_board;


struct SearchPathResult
{
    vector<Pos> path;
    vector<Pos> need_pos;
    vector<Pos> obs_pos;

    bool is_valid() const
    {
        return path.size();
    }

    bool need_extra_moves() const
    {
        return need_pos.size() + obs_pos.size();
    }

    string size_info() const
    {
        char buf[128];
        sprintf(buf, "%3d %3d %3d", (int)path.size(), (int)need_pos.size(), (int)obs_pos.size());
        return buf;
    }
};
SearchPathResult search_ball_path(Board board, const Pos& start_pos, const vector<bool>& allow_colors)
{
    assert(!board.is_wall(start_pos));
    assert(!board.is_color(start_pos));
    assert(count(all(allow_colors), true));

    const int inf = 1919810;
    int dp[64][64][4];
    int prev_dir[64][64][4];
    rep(y, board.height()) rep(x, board.width()) rep(dir, 4)
        dp[y][x][dir] = inf;

    using P = tuple<int, Pos, int>;
    priority_queue<P, vector<P>, greater<P>> q;

#if 1
    const int NEW_NEED_COST = 10;
    const int MATCH_MOVE_COST = 5;
    const int REMOVE_COST = 50;
#else
    const int NEW_NEED_COST = 50000;
    const int MATCH_MOVE_COST = 10000;
    const int REMOVE_COST = 30000;
#endif
    rep(dir, 4)
    {
        dp[start_pos.y][start_pos.x][dir] = 0;
        prev_dir[start_pos.y][start_pos.x][dir] = -1;
    }
    rep(dir, 4)
    {
        const Pos next = start_pos.next(dir);
        if (!board.is_wall(next) && (!board.is_color(next) || allow_colors[board.color(next)]))
        {
            int cost = 1;
            if (!board.is_obs(start_pos.next((dir + 2) & 3)))
                cost += NEW_NEED_COST;
            if (board.is_color(next) && board.color(next) == target_board.color(next))
                cost += MATCH_MOVE_COST;
            if (board.is_color(next) && !allow_colors[board.color(next)])
                cost += REMOVE_COST;

            dp[next.y][next.x][dir] = cost;
            prev_dir[next.y][next.x][dir] = dir;
            q.push(P(cost, next, dir));
        }
    }

    while (!q.empty())
    {
        int cost, cur_dir;
        Pos cur;
        tie(cost, cur, cur_dir) = q.top();
        q.pop();

        if (cost > 30)
            return SearchPathResult();

        if (cost > dp[cur.y][cur.x][cur_dir])
            continue;

        if (board.is_color(cur) && allow_colors[board.color(cur)])
        {
//             cerr << endl;
//             dump(start_pos);
//             dump(cur);
//             dump(dp[cur.y][cur.x][cur_dir]);

            assert(allow_colors[board.color(cur)]);

            vector<Pos> path = {cur};
            vector<Pos> need_pos, obs_pos;
            Pos p = cur;
            int d = cur_dir;
            while (true)
            {
                assert(0 <= d && d < 4);

                const int pd = prev_dir[p.y][p.x][d];

//                 cerr << "-------------------------------" << endl;
//                 dump(p);
//                 dump(S_DIR[d]);
//                 dump(dp[p.y][p.x][d]);
//                 dump((0 <= pd && pd < 4 ? S_DIR[prev_dir[p.y][p.x][d]] : '*'));

                if (pd == -1)
                    break;

                p.move((d + 2) & 3);

                if (pd != d)
                {
                    path.push_back(p);
//
                    Pos stopper = p.next((d + 2) & 3);
                    if (!board.is_obs(stopper))
                        need_pos.push_back(stopper);
//
// //                     assert(board.is_obs(stopper)); // if no need_pos
                }
                if (board.is_color(p) && p != start_pos)
                    obs_pos.push_back(p);

                d = pd;
            }
            path.push_back(start_pos);
            if (!board.is_obs(start_pos.next(moved_dir(path[path.size() - 2], path.back()))))
            {
                need_pos.push_back(start_pos.next(moved_dir(path[path.size() - 2], path.back())));
            }
//             dump(path);

            bool ok = true;
            rep(i, (int)path.size() - 1)
            {
                assert(path[i] != path[i + 1]);
                if (path[i + 1].next(moved_dir(path[i], path[i + 1])) == path[0])
                {
                    ok = false;
                    break;
                }
            }

            if (ok)
            {
                return SearchPathResult {
                    path,
                    need_pos,
                    obs_pos,
                };
            }
        }
        else
        {
            rep(dir, 4)
            {
//                 if (dir != cur_dir && !board.is_obs(cur.next((dir + 2) & 3)))
//                     continue;

                const Pos next = cur.next(dir);
                if (!board.is_wall(next))
                {
                    int ncost = cost + (dir == cur_dir ? 0 : 1);
                    if (dir != cur_dir && !board.is_obs(cur.next((dir + 2) & 3)))
                        ncost += NEW_NEED_COST;
                    if (board.is_color(next) && board.color(next) == target_board.color(next))
                        ncost += MATCH_MOVE_COST;
                    if (board.is_color(next) && !allow_colors[board.color(next)])
                        ncost += REMOVE_COST;

                    if (ncost < dp[next.y][next.x][dir])
                    {
                        dp[next.y][next.x][dir] = ncost;
                        prev_dir[next.y][next.x][dir] = cur_dir;
                        q.push(P(ncost, next, dir));
                    }
                }
            }
        }
    }
    return SearchPathResult();
}

class RollingBalls
{
public:
    vector<string> restorePattern(const vector<string>& start, const vector<string>& target)
    {
        ::target_board = Board(target);
        Board board(start);
        const int h = target_board.height(), w = target_board.width();

        vector<Pos> target_poss;
        rep(y, h) rep(x, w)
        {
            if (target_board.is_color(x, y))
                target_poss.push_back(Pos(x, y));
        }

        const int max_rolls = target_poss.size() * 20;

        vector<string> res;
        int last_try_i = -1;
        rep(try_i, max_rolls * 10)
        {
            if (try_i - last_try_i > max_rolls * 4)
                break;

            vector<Pos> unmatch_target_poss;
            rep(y, h) rep(x, w)
            {
                if (target_board.is_color(x, y) && !board.is_color(x, y))
                {
                    unmatch_target_poss.push_back(Pos(x, y));

                    if (try_i > 2 * max_rolls)
                    {
                        rep(dir, 4)
                        {
                            Pos q = Pos(x, y).next(dir);
                            if (board.empty(q))
                            {
                                unmatch_target_poss.push_back(q);
                            }
                        }
                    }
                }
            }
            if (unmatch_target_poss.empty())
                break;

            Pos target_pos = unmatch_target_poss[g_rand.next_int(unmatch_target_poss.size())];
//             Pos target_pos = unmatch_target_poss[try_i % unmatch_target_poss.size()];
//             if (try_i > 3000)
//                 target_pos = Pos(3, 0);

            vector<bool> allow_colors(20);
            allow_colors[target_board.color(target_pos)] = true;
//             if (try_i > 2 * max_rolls)
//             {
//                 rep(i, 20)
//                     allow_colors[i] = true;
//             }
            auto result = search_ball_path(board, target_pos, allow_colors);
            if (!result.is_valid())
                continue;

//             if (result.need_extra_moves())
//             {
//                 dump(target_pos);
//                 dump(try_i);
//                 dump(result.size_info());
//                 dump(result.path);
// //                 continue;
//             }

            Board nboard = board;
            vector<string> nres = res;

            bool ex = false;
            if (result.need_pos.size() > 0 && result.obs_pos.empty())
            {
                auto recur_result = search_ball_path(nboard, result.need_pos[0], vector<bool>(20, true));
//                 dump(recur_result.size_info());
//                 dump(recur_result.path);
                if (!recur_result.is_valid() || recur_result.need_extra_moves())
                    continue;

                if (res.size() + recur_result.path.size() - 1 > max_rolls)
                    break;

                nboard.move(recur_result.path[0], recur_result.path.back());
                add_res(nres, recur_result.path);

                ex = true;
//                 cerr << "extra" << endl;
            }

            result = search_ball_path(nboard, target_pos, allow_colors);
            if (!result.is_valid() || result.need_extra_moves())
                continue;

            const auto& path = result.path;
            if (nres.size() + path.size() - 1 > max_rolls)
                break;

//             dump(path);
            nboard.move(path[0], path.back());
            add_res(nres, path);
            if (ex)
            {
//                 cerr << "extra match" << endl;
            }

            board = nboard;
            res = nres;


            last_try_i = try_i;
        }

        dump(board.game_score(target_board));

        return res;
    }

private:
    void add_res(vector<string>& res, const vector<Pos>& path)
    {
        assert(path.size() >= 2);

        rep(i, path.size() - 1)
        {
            char buf[128];
            int dir = moved_dir(path[i], path[i + 1]);
            sprintf(buf, "%d %d %d", path[i].y, path[i].x, dir);
            res.push_back(buf);
        }
    }
};



#ifdef LOCAL
int main()
{
    int h;
    cin >> h;
    vector<string> start(h), target(h);
    input(start, h);
    cin >> h;
    assert(h == start.size());
    input(target, h);

    auto ret = RollingBalls().restorePattern(start, target);
    cout << ret.size() << endl;
    for (auto& s : ret)
        cout << s << endl;
    cout.flush();
}
#endif
