#include <bits/stdc++.h>
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
void solve() {
    int n; 
        cin>>n;
        vector<int>arr(n+1), prem(n+1), suffm(n+2);
        for(int i=1;i<=n;i++) 
            cin>>arr[i];

  
        prem[1]=arr[1];
        for(int i=2;i<=n;i++)
            prem[i]=min(prem[i-1], arr[i]);


        suffm[n]=arr[n];
        for(int i=n-1;i>=1;i--)
            suffm[i]=max(suffm[i+1], arr[i]);


        for(int i=1;i<=n;i++){
            if(prem[i]==arr[i] || suffm[i]==arr[i])
                cout<<'1';
            else
                cout<<'0';
        }
        cout<<"\n";
}
int main(){
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    int t; 
    cin>>t;
    while(t--){
        solve();
    }
    return 0;
}