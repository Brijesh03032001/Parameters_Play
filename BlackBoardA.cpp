#include <bits/stdc++.h>
using namespace std;
void solve()
{
    int n;cin>>n;

    int quo=n/4,rem=n%4;
    int c[4];
        for(int i=0;i<4;i++){
            c[i]=quo+(rem>i);
        }

        int total_pairs=min(c[0],c[3])+min(c[1],c[2]);
       
        if (total_pairs * 2 == n) {
            std::cout << "Bob\n";
        } else {
            std::cout << "Alice\n";
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
int main(){
    int test;
    cin>>test;
    while(test--){
        solve();
    }
    return 0;
}

