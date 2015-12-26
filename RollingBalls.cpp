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
Random g_rand(9);

struct ZobristHash
{
    ZobristHash()
    {
        mt19937_64 mt;
        rep(y, 64) rep(x, 64) rep(c, 13)
            h[y][x][c] = mt();
    }

    ull h[64][64][13];
};
const ZobristHash zobrist;

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

    ull hash() const
    {
        ull ha = 0;
        rep(y, h) rep(x, w)
            ha ^= zobrist.h[y][x][color(x, y)];
        return ha;
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
set<tuple<ull, int, Pos, Pos>> taboo;
SearchPathResult search_ball_path(const Board& board, const Pos& start_pos, const vector<bool>& allow_colors, const vector<vector<int>>& match_cost)
{
    assert(!board.is_wall(start_pos));
    assert(!board.is_color(start_pos));
    assert(board.empty(start_pos));
    assert(count(all(allow_colors), true));

    const int inf = 100;
    int dp[64][64][4];
    int prev_dir[64][64][4];
    int goal_dp[64][64][4];
    int goal_prev_dir[64][64][4];
    rep(y, board.height()) rep(x, board.width()) rep(dir, 4)
    {
        dp[y][x][dir] = inf;
        goal_dp[y][x][dir] = inf;
    }

    using P = tuple<int, Pos, int, bool>;
    priority_queue<P, vector<P>, greater<P>> q;

#if 1
    const int NEW_NEED_COST = 5;
    const int MATCH_MOVE_COST = 10;
    const int UNEXPECT_RETURN_COST = 0;
    const int REMOVE_COST = 3000;
#else
    const int NEW_NEED_COST = 50000;
    const int MATCH_MOVE_COST = 10000;
    const int UNEXPECT_RETURN_COST = 200000;
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

            if (board.is_color(next))
            {
                const bool easy_return = board.is_obs(next.next(dir));
                if (!easy_return)
                    cost += UNEXPECT_RETURN_COST;

                if (allow_colors[board.color(next)])
                {
                    int gcost = cost;
                    if (!easy_return)
                        gcost += UNEXPECT_RETURN_COST;
                    if (board.color(next) == target_board.color(next))
                    {
                        if (easy_return)
                            gcost += MATCH_MOVE_COST;
                        else
                            gcost += 2 * MATCH_MOVE_COST;
                    }
                    gcost += match_cost[next.y][next.x];

                    goal_dp[next.y][next.x][dir] = gcost;
                    goal_prev_dir[next.y][next.x][dir] = -1;
                    q.push(P(gcost, next, dir, true));
                }

                cost += REMOVE_COST;
            }


            dp[next.y][next.x][dir] = cost;
            prev_dir[next.y][next.x][dir] = -1;
            q.push(P(cost, next, dir, false));
        }
    }

    const ull board_hash = board.hash();

    while (!q.empty())
    {
        int cost, cur_dir;
        Pos cur;
        bool is_goal;
        tie(cost, cur, cur_dir, is_goal) = q.top();
        q.pop();

        assert(!board.is_wall(cur));
        assert(0 <= cur_dir && cur_dir < 4);
        assert(in_rect(cur.x, cur.y, board.width(), board.height()));

        if (cost > 50)
            return SearchPathResult();

        if (cost > dp[cur.y][cur.x][cur_dir])
            continue;

        if (is_goal)
        {
            assert(board.is_color(cur) && allow_colors[board.color(cur)]);
            auto key = make_tuple(board_hash, cost, start_pos, cur);
            if (taboo.count(key))
                continue;
            taboo.insert(key);

//             cerr << endl;
//             dump(start_pos);
//             dump(cur);
//             dump(cost);
//             dump(dp[cur.y][cur.x][cur_dir]);

            assert(allow_colors[board.color(cur)]);

            vector<Pos> path = {cur};
            vector<Pos> need_pos, obs_pos;
            Pos p = cur.next((cur_dir + 2) & 3);
            int d = goal_prev_dir[cur.y][cur.x][cur_dir];
            if (d != -1 && d != cur_dir)
            {
                path.push_back(p);
                //
                Pos stopper = p.next((cur_dir + 2) & 3);
                if (!board.is_obs(stopper))
                    need_pos.push_back(stopper);
            }
            while (d != -1)
            {
                assert(0 <= d && d < 4);
                assert(in_rect(p.x, p.y, board.width(), board.height()));

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
                if (board.is_color(p) && p != cur)
                {
//                     dump(start_pos);
//                     dump(p);
//                     dump(cur);
//                     dump(path);
//                     dump(cost);
//                     assert(false);
                    obs_pos.push_back(p);
                }

                d = pd;
            }
            path.push_back(start_pos);
            if (!board.is_obs(start_pos.next(moved_dir(path[path.size() - 2], path.back()))))
            {
                need_pos.push_back(start_pos.next(moved_dir(path[path.size() - 2], path.back())));
            }
//             dump(path);
            assert(obs_pos.empty());

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
            assert(!is_goal);

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

                    if (board.is_color(next))
                    {
                        const bool easy_return = board.is_obs(next.next(dir));

                        if (allow_colors[board.color(next)])
                        {
                            int gcost = ncost;
                            if (!easy_return)
                                gcost += UNEXPECT_RETURN_COST;
                            if (board.color(next) == target_board.color(next))
                            {
                                if (easy_return)
                                    gcost += MATCH_MOVE_COST;
                                else
                                    gcost += 2 * MATCH_MOVE_COST;
                            }
                            gcost += match_cost[next.y][next.x];

                            if (gcost < goal_dp[next.y][next.x][dir])
                            {
                                goal_dp[next.y][next.x][dir] = gcost;
                                goal_prev_dir[next.y][next.x][dir] = cur_dir;
                                q.push(P(gcost, next, dir, true));
                            }
                        }

                        ncost += REMOVE_COST;
                    }


                    if (ncost < dp[next.y][next.x][dir])
                    {
                        dp[next.y][next.x][dir] = ncost;
                        prev_dir[next.y][next.x][dir] = cur_dir;
                        q.push(P(ncost, next, dir, false));
                    }
                }
            }
        }
    }
    return SearchPathResult();
}

vector<Pos> search_path_to_remove_ball(const Board& board, const Pos& start_pos, const vector<vector<int>>& goal_cost)
{
    const int inf = 1000;
    int dp[64][64];
    Pos prev_pos[64][64];
    rep(y, board.height()) rep(x, board.width())
        dp[y][x] = inf;

    using P = tuple<int, bool, Pos>;
    priority_queue<P, vector<P>, greater<P>> q;
    dp[start_pos.y][start_pos.x] = 0;
    q.push(P(0, false, start_pos));
    while (!q.empty())
    {
        int cost;
        bool is_goal;
        Pos cur;
        tie(cost, is_goal, cur) = q.top();
        q.pop();

        if (is_goal)
        {
            vector<Pos> path;
            for (Pos p = cur; p != start_pos; p = prev_pos[p.y][p.x])
                path.push_back(p);
            path.push_back(start_pos);
            reverse(all(path));
            return path;
        }

        rep(dir, 4)
        {
            Pos next = cur.next(dir);
            if (board.is_obs(next))
                continue;

            while (board.empty(next))
                next.move(dir);
            next.move((dir + 2) & 3);

            // TODO: WALLとcolorでコストに差をつける
            int ncost = cost + 1;
            const int UNEXPECT_RETURN_COST = 10;
            if (board.empty(cur.next((dir + 2) & 3)))
                ncost += UNEXPECT_RETURN_COST;

            if (ncost < dp[next.y][next.x])
            {
                dp[next.y][next.x] = ncost;
                q.push(P(ncost, false, next));
            }

            int gcost = ncost + goal_cost[next.y][next.x];
            if (gcost < inf)
                q.push(P(gcost, true, next));
        }
    }
    return {};
}

vector<vector<Pos>> search_paths(Board board, const Pos& main_target)
{
    const int h = board.height(), w = board.width();

    vector<vector<Pos>> paths;
    // TODO: stackでやってるのは、ダブリdijkstraを避けるため。stackなくすべきか？
    vector<Pos> target_stack;
    target_stack.push_back(main_target);
    vector<vector<int>> match_cost(h, vector<int>(w));
    set<pair<int, vector<Pos>>> visited;
    while (!target_stack.empty())
    {
//         dump(make_pair(target_stack.size(), paths.size()));
        auto key = make_pair(paths.size(), target_stack);
        if (visited.count(key))
            return {};
        visited.insert(key);

        int moves = 0;
        for (auto& path : paths)
            moves += path.size() - 1;
        if (moves > 20)
            return {};

        if (target_stack.size() >= 5)
            return {};
        if (paths.size() >= 5)
            return {};

        Pos target = target_stack.back();
        target_stack.pop_back();
        assert(count(all(target_stack), target) == 0);
        assert(board.empty(target));

        vector<bool> allow_colors(16);
        if (target_stack.empty())
            allow_colors[target_board.color(target)] = true;
        else
//             fill_n(allow_colors.begin(), WALL, true);
            fill(all(allow_colors), true);

        SearchPathResult result = search_ball_path(board, target, allow_colors, match_cost);

//         if (target_stack.empty() && main_target == Pos(12, 16))
//             dump(result.path);

        if (!result.is_valid())
            return {};
        if (result.obs_pos.size())
            return {};

        if (result.need_pos.size())
        {
            Pos need_target = result.need_pos.back();

            auto it = find(all(target_stack), need_target);
            if (it != target_stack.end())
                target_stack.erase(it);

            target_stack.push_back(target);
            target_stack.push_back(need_target);

//         dump(result.size_info());
//             dump(target_stack);
        }
        else
        {
            assert(!result.need_extra_moves());

            paths.push_back(result.path);
            board.move(result.path[0], result.path.back());
            match_cost[result.path.back().y][result.path.back().x] += 15;
        }
    }
    return paths;
}

class RollingBalls
{
public:
    vector<string> restorePattern(const vector<string>& start, const vector<string>& target)
    {
        ::target_board = Board(target);
        Board start_board(start);
        const int h = target_board.height(), w = target_board.width();

        vector<Pos> target_poss;
        rep(y, h) rep(x, w)
        {
            if (target_board.is_color(x, y))
                target_poss.push_back(Pos(x, y));
        }

        const int max_rolls = target_poss.size() * 20;

        double best_score = -1;
        vector<string> best_res;

        rep(trytry_i, 100)
        {
            taboo.clear();
            Board board = start_board;

            vector<string> res;
            int last_try_i = -1;
            rep(try_i, max_rolls * 10)
            {
                if (try_i - last_try_i > min<int>(max_rolls, 2 * target_poss.size()))
                    break;

                vector<Pos> unmatch_target_poss;
                rep(y, h) rep(x, w)
                {
                    if (target_board.is_color(x, y) && !board.is_color(x, y))
                    {
                        unmatch_target_poss.push_back(Pos(x, y));
                    }
                }
                if (unmatch_target_poss.empty())
                    break;

                Pos target_pos = unmatch_target_poss[g_rand.next_int(unmatch_target_poss.size())];
//                 Pos target_pos = unmatch_target_poss[(trytry_i * 10 + try_i) % unmatch_target_poss.size()];
                //             if (try_i > 500 && board.empty(Pos(12, 16)))
                //                 target_pos = Pos(12, 16);

                auto paths = search_paths(board, target_pos);
                if (paths.empty())
                    continue;

                Board nboard = board;
                vector<string> nres = res;
                for (auto& path : paths)
                {
                    nboard.move(path[0], path.back());
                    add_res(path, nres);
                }
                if (nres.size() > max_rolls)
                    continue;

                double score = nboard.game_score(target_board);
                if (score > best_score)
                {
                    best_score = score;
                    best_res = nres;
                }

                board = nboard;
                res = nres;

                last_try_i = try_i;
            }
        }
        dump(best_score);
        return best_res;
    }

private:
    void add_res(const vector<Pos>& path, vector<string>& res)
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
