#include <iostream>
#include <vector>
#include <cstdio>
  #include <iostream>
  #include <map>
  #include <string>
  #include <unordered_map>
  #include <unordered_set>
  #include <set>
  #include <vector>
  #include <stack>
  
  const long long INF = 9223372036854775807;
  const long long MOD = 1e9 + 7;
#include <queue>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <bitset>
using namespace std;
void solve(){
    int n, jo, koo;
    cin >> n 
    cin >> jo >> koo;
    vector<int> arr(n);
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }
    int x = arr[jo-1];

    long long g1 = 0, g2 = 0, g3 = 0;
    for (int i = 0; i < n; i++) {
        if (arr[i] > x) g1++;
        else if (arr[i] < x) g3++;
        else g2++;
    }

    long long remem = 0;
    if (g1 > 0) {
        remem += (g1 - 1);
    }
    remem += (g2 - 1);
    remem += g3;

    if (rem >= n - koo) {
        cout << "YES\n";
    } else {
        cout << "NO\n";
    }
    return ;
}
vector<int> topo(int N, vector<vector<int>>& adj)
      {
          vector<int>ind(N,0);
          for(int u=0;u<N;u++)
          {
              for(auto it : adj[u])
              {
                  ind[it]++;
              }
          }
          
          queue<int>qu;
          for(int v = 0; v<N;v++)
          {
              if(ind[v] == 0)
              {
                  qu.push(v);
              }
          }
          vector<int> res ;
          while(!qu.empty())
          {
              int src = qu.front();
              res.push_back(src);
              qu.pop();
              for(auto it : adj[src])
              {
                  ind[it]--;
                  if(ind[it] == 0)
                  {
                      qu.push(it);
                  }
              }
          }
          return res;
      }
int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int t;
    cin >> t;
    while (t--) {
       solve();
    }
    return 0;
}


