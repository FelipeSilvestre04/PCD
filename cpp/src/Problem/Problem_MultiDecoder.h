#ifndef _PROBLEM_MULTIDECODER_H
#define _PROBLEM_MULTIDECODER_H

// ============================================================================
// HYPER-HEURISTIC MULTI-DECODER
//
// Chromosome: N + 2
//   [0..N-1]:  Product Priorities (ordering for RCL)
//   [N]:       Alpha (greediness: K = max(1, ceil(α * remaining)))
//   [N+1]:     Heuristic Selector (discretized to 5 bands)
//
// Heuristics:
//   H0: GCI      — Global Cheapest Insertion (baseline)
//   H1: Balanced  — λ*delta_cost + (1-λ)*norm_load, λ=alpha
//   H2: LookAhead — GCI + micro-swaps after each insertion
//   H3: FixOpt    — batch K products, test K! permutations
//   H4: ATCS      — Apparient Tardiness Cost with Setups (append-only)
// ============================================================================

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <omp.h>

struct TProblemData {
    int n;
    int num_products;
    int num_machines;
    std::vector<double> machine_capacities;
    std::vector<int> initial_state;
    std::vector<double> demands;
    std::vector<double> production_rates;
    std::vector<double> setup_costs;
    std::vector<double> setup_times;
    double start_time = 0.0;
    double max_time = 1e9;
};

void ReadData(char name[], TProblemData &data) {
    std::ifstream file(name);
    if (!file.is_open()) { printf("\nERROR: File (%s) not found!\n", name); exit(1); }
    file >> data.num_products >> data.num_machines;
    data.n = data.num_products + 2; // N + 2
    int np = data.num_products, nm = data.num_machines;
    data.machine_capacities.resize(nm);
    for(int i=0;i<nm;i++) file >> data.machine_capacities[i];
    data.initial_state.resize(nm);
    for(int i=0;i<nm;i++) file >> data.initial_state[i];
    data.demands.resize(np);
    for(int i=0;i<np;i++) file >> data.demands[i];
    data.production_rates.resize(np*nm);
    for(int i=0;i<np;i++) for(int m=0;m<nm;m++) file >> data.production_rates[i*nm+m];
    data.setup_costs.resize(np*np*nm);
    std::string label;
    for(int m=0;m<nm;m++){file>>label;for(int i=0;i<np;i++)for(int j=0;j<np;j++){
        int idx=i*np*nm+j*nm+m; file>>data.setup_costs[idx];}}
    data.setup_times.resize(np*np*nm);
    for(int m=0;m<nm;m++){file>>label;for(int i=0;i<np;i++)for(int j=0;j<np;j++){
        int idx=i*np*nm+j*nm+m; file>>data.setup_times[idx];}}
    file.close();
}

// ==================== helpers ====================
static inline int IX(int i,int j,int m,int np,int nm){return i*np*nm+j*nm+m;}

struct InsDelta{double dc,dt;};
static InsDelta calcDelta(int p,int m,int pos,
    const std::vector<std::vector<int>>&sq,const std::vector<double>&ld,const TProblemData&d){
    int np=d.num_products,nm=d.num_machines;
    const auto&s=sq[m]; int sl=(int)s.size();
    int pv=(pos==0)?d.initial_state[m]:s[pos-1];
    int nx=(pos<sl)?s[pos]:-1;
    double r=d.production_rates[p*nm+m],pt=(r>1e-6)?d.demands[p]/r:1e9;
    double ca=d.setup_costs[IX(pv,p,m,np,nm)],cr=0;
    double ta=d.setup_times[IX(pv,p,m,np,nm)],tr=0;
    if(nx!=-1){ca+=d.setup_costs[IX(p,nx,m,np,nm)];cr=d.setup_costs[IX(pv,nx,m,np,nm)];
               ta+=d.setup_times[IX(p,nx,m,np,nm)];tr=d.setup_times[IX(pv,nx,m,np,nm)];}
    return {ca-cr,ta+pt-tr};
}

// Find best (product,machine,position) among candidates × machines
struct BestIns{int p,m,pos; double cost,time; bool found;};
static BestIns findBestInsertion(const std::vector<int>&cands,
    const std::vector<std::vector<int>>&sq,const std::vector<double>&ld,const TProblemData&d){
    int nm=d.num_machines; BestIns b={-1,-1,-1,1e18,0,false};
    for(int p:cands) for(int m=0;m<nm;m++){
        if(d.production_rates[p*nm+m]<=1e-6) continue;
        for(int pos=0;pos<=(int)sq[m].size();pos++){
            auto info=calcDelta(p,m,pos,sq,ld,d);
            if(ld[m]+info.dt<=d.machine_capacities[m]&&info.dc<b.cost)
            {b={p,m,pos,info.dc,info.dt,true};}
        }
    }
    return b;
}

// ====================================================================
// H0: GCI — Global Cheapest Insertion
// ====================================================================
static double H0_GCI(const std::vector<int>&sorted, double alpha, const TProblemData&d){
    int np=d.num_products,nm=d.num_machines;
    std::vector<std::vector<int>>sq(nm); std::vector<double>ld(nm,0);
    std::vector<bool>al(np,false); int done=0; double tot=0,pen=0;
    while(done<np){
        int rem=np-done, k=std::max(1,(int)std::ceil(alpha*rem));
        std::vector<int>cands; cands.reserve(k);
        for(int p:sorted) if(!al[p]){cands.push_back(p);if((int)cands.size()==k)break;}
        auto b=findBestInsertion(cands,sq,ld,d);
        if(b.found){sq[b.m].insert(sq[b.m].begin()+b.pos,b.p);ld[b.m]+=b.time;tot+=b.cost;al[b.p]=true;done++;}
        else{pen+=1e9+d.demands[cands[0]]*1000;al[cands[0]]=true;done++;}
    }
    return tot+pen;
}

// ====================================================================
// H1: Balanced CI — Score = λ*delta_cost + (1-λ)*norm_load*100
// ====================================================================
static double H1_Balanced(const std::vector<int>&sorted, double alpha, const TProblemData&d){
    int np=d.num_products,nm=d.num_machines;
    double lambda=alpha; // alpha controls balance: low=cost focus, high=load balance
    double mcap=*std::max_element(d.machine_capacities.begin(),d.machine_capacities.end());
    std::vector<std::vector<int>>sq(nm); std::vector<double>ld(nm,0);
    std::vector<bool>al(np,false); int done=0; double tot=0,pen=0;
    while(done<np){
        int rem=np-done, k=std::max(1,(int)std::ceil(alpha*rem));
        std::vector<int>cands; cands.reserve(k);
        for(int p:sorted)if(!al[p]){cands.push_back(p);if((int)cands.size()==k)break;}
        int bp=-1,bm=-1,bpos=-1;double bsc=1e18,bc=0,bt=0;
        for(int p:cands)for(int m=0;m<nm;m++){
            if(d.production_rates[p*nm+m]<=1e-6)continue;
            double nl=ld[m]/mcap;
            for(int pos=0;pos<=(int)sq[m].size();pos++){
                auto info=calcDelta(p,m,pos,sq,ld,d);
                if(ld[m]+info.dt<=d.machine_capacities[m]){
                    double sc=lambda*info.dc+(1.0-lambda)*nl*100.0;
                    if(sc<bsc){bsc=sc;bc=info.dc;bt=info.dt;bp=p;bm=m;bpos=pos;}
                }
            }
        }
        if(bp!=-1){sq[bm].insert(sq[bm].begin()+bpos,bp);ld[bm]+=bt;tot+=bc;al[bp]=true;done++;}
        else{pen+=1e9+d.demands[cands[0]]*1000;al[cands[0]]=true;done++;}
    }
    return tot+pen;
}

// ====================================================================
// H2: LookAhead — GCI + micro-swaps on affected machine
// ====================================================================
static double H2_LookAhead(const std::vector<int>&sorted, double alpha, const TProblemData&d){
    int np=d.num_products,nm=d.num_machines;
    std::vector<std::vector<int>>sq(nm); std::vector<double>ld(nm,0);
    std::vector<bool>al(np,false); int done=0; double tot=0,pen=0;

    auto reLoad=[&](int m){double l=0;int pv=d.initial_state[m];
        for(int p:sq[m]){l+=d.setup_times[IX(pv,p,m,np,nm)]+d.demands[p]/d.production_rates[p*nm+m];pv=p;}return l;};
    auto reCost=[&](int m){double c=0;int pv=d.initial_state[m];
        for(int p:sq[m]){c+=d.setup_costs[IX(pv,p,m,np,nm)];pv=p;}return c;};

    while(done<np){
        int rem=np-done, k=std::max(1,(int)std::ceil(alpha*rem));
        std::vector<int>cands; cands.reserve(k);
        for(int p:sorted)if(!al[p]){cands.push_back(p);if((int)cands.size()==k)break;}
        auto b=findBestInsertion(cands,sq,ld,d);
        if(b.found){
            sq[b.m].insert(sq[b.m].begin()+b.pos,b.p);ld[b.m]+=b.time;al[b.p]=true;done++;
            // Micro-swaps
            bool imp=true;
            while(imp){imp=false;double cb=reCost(b.m);
                for(int i=0;i<(int)sq[b.m].size()-1;i++){
                    std::swap(sq[b.m][i],sq[b.m][i+1]);
                    double nl=reLoad(b.m);
                    if(nl<=d.machine_capacities[b.m]){double nc=reCost(b.m);
                        if(nc<cb){cb=nc;ld[b.m]=nl;imp=true;}
                        else std::swap(sq[b.m][i],sq[b.m][i+1]);}
                    else std::swap(sq[b.m][i],sq[b.m][i+1]);
                }
            }
        } else{pen+=1e9+d.demands[cands[0]]*1000;al[cands[0]]=true;done++;}
    }
    tot=0;for(int m=0;m<nm;m++)tot+=reCost(m);
    return tot+pen;
}

// ====================================================================
// H3: Fix-Optimize — batch K products, try K! permutations
// ====================================================================
static double H3_FixOpt(const std::vector<int>&sorted, double alpha, const TProblemData&d){
    int np=d.num_products,nm=d.num_machines;
    int batch_k=2+(int)(alpha*3); // alpha→{2,3,4}
    if(batch_k>4)batch_k=4; if(batch_k<2)batch_k=2;
    std::vector<std::vector<int>>sq(nm); std::vector<double>ld(nm,0);
    std::vector<bool>al(np,false); int done=0; double tot=0,pen=0;

    while(done<np){
        int rem=np-done;
        int bk=std::min(batch_k,rem);
        // Collect batch from sorted order
        std::vector<int>batch; batch.reserve(bk);
        for(int p:sorted)if(!al[p]){batch.push_back(p);if((int)batch.size()==bk)break;}

        if(bk<=1){
            auto b=findBestInsertion(batch,sq,ld,d);
            if(b.found){sq[b.m].insert(sq[b.m].begin()+b.pos,b.p);ld[b.m]+=b.time;tot+=b.cost;}
            else{pen+=1e9;} al[batch[0]]=true;done++;continue;
        }
        // Try all permutations
        std::vector<int>perm=batch; std::sort(perm.begin(),perm.end());
        double bestC=1e18; std::vector<std::vector<int>>bestSq; std::vector<double>bestLd;
        do{
            auto tSq=sq;auto tLd=ld;double tC=tot;bool ok=true;
            for(int p:perm){
                auto b=findBestInsertion({p},tSq,tLd,d);
                if(b.found){tSq[b.m].insert(tSq[b.m].begin()+b.pos,b.p);tLd[b.m]+=b.time;tC+=b.cost;}
                else{ok=false;tC+=1e9;break;}
            }
            if(tC<bestC){bestC=tC;bestSq=tSq;bestLd=tLd;}
        }while(std::next_permutation(perm.begin(),perm.end()));
        sq=bestSq;ld=bestLd;tot=bestC;
        for(int p:batch){al[p]=true;done++;}
    }
    return tot+pen;
}

// ====================================================================
// H4: ATCS — Apparent Tardiness Cost with Setups (append-only)
// ====================================================================
static double H4_ATCS(const std::vector<int>&sorted, double alpha, const TProblemData&d){
    int np=d.num_products,nm=d.num_machines;
    const double eps=0.01;
    std::vector<std::vector<int>>sq(nm); std::vector<double>ld(nm,0);
    std::vector<bool>al(np,false); int done=0; double tot=0,pen=0;

    while(done<np){
        int rem=np-done, k=std::max(1,(int)std::ceil(alpha*rem));
        std::vector<int>cands; cands.reserve(k);
        for(int p:sorted)if(!al[p]){cands.push_back(p);if((int)cands.size()==k)break;}

        int bp=-1,bm=-1;double bsc=-1e18,bc=0,bt=0;
        for(int p:cands){
            // Urgency = position in sorted (earlier = more urgent)
            double urgency=1.0;
            for(int i=0;i<(int)sorted.size();i++)if(sorted[i]==p){urgency=1.0-(double)i/np;break;}
            for(int m=0;m<nm;m++){
                double r=d.production_rates[p*nm+m];if(r<=1e-6)continue;
                double pt=d.demands[p]/r;
                int last=sq[m].empty()?d.initial_state[m]:sq[m].back();
                double sc=d.setup_costs[IX(last,p,m,np,nm)];
                double st=d.setup_times[IX(last,p,m,np,nm)];
                if(ld[m]+st+pt<=d.machine_capacities[m]){
                    double score=(d.demands[p]/(sc+eps))*urgency;
                    if(score>bsc){bsc=score;bc=sc;bt=st+pt;bp=p;bm=m;}
                }
            }
        }
        if(bp!=-1){sq[bm].push_back(bp);ld[bm]+=bt;tot+=bc;al[bp]=true;done++;}
        else{pen+=1e9+d.demands[cands[0]]*1000;al[cands[0]]=true;done++;}
    }
    return tot+pen;
}

// ====================================================================
// MAIN DECODER
// ====================================================================
double Decoder(TSol &s, const TProblemData &data){
    int np=data.num_products;
    double alpha=s.rk[np];     // gene N
    double h_sel=s.rk[np+1];   // gene N+1

    // Sort products by keys
    std::vector<int>sorted(np);
    std::iota(sorted.begin(),sorted.end(),0);
    std::sort(sorted.begin(),sorted.end(),[&](int a,int b){return s.rk[a]<s.rk[b];});

    // Select heuristic
    int h=(int)(h_sel*5.0); if(h>4)h=4;

    switch(h){
        case 0: return H0_GCI(sorted,alpha,data);
        case 1: return H1_Balanced(sorted,alpha,data);
        case 2: return H2_LookAhead(sorted,alpha,data);
        case 3: return H3_FixOpt(sorted,alpha,data);
        case 4: return H4_ATCS(sorted,alpha,data);
        default:return H0_GCI(sorted,alpha,data);
    }
}

void FreeMemoryProblem(TProblemData &data){
    data.machine_capacities.clear();data.initial_state.clear();
    data.demands.clear();data.production_rates.clear();
    data.setup_costs.clear();data.setup_times.clear();
}

void PrintSolution(TSol &s, const TProblemData &d){
    int np=d.num_products,nm=d.num_machines;
    double alpha=s.rk[np]; double h_sel=s.rk[np+1];
    int h=(int)(h_sel*5.0); if(h>4)h=4;
    const char*names[]={"GCI","Balanced","LookAhead","FixOpt","ATCS"};
    printf("\n[MultiDecoder H=%d(%s) Alpha=%.4f]\n",h,names[h],alpha);

    std::vector<int>sorted(np);
    std::iota(sorted.begin(),sorted.end(),0);
    std::sort(sorted.begin(),sorted.end(),[&](int a,int b){return s.rk[a]<s.rk[b];});
    std::vector<std::vector<int>>sq(nm); std::vector<double>ld(nm,0);
    std::vector<bool>al(np,false); int done=0;

    if(h==0 || h==2){ // GCI & LookAhead (printing lookahead swaps is complex, we just print the basic GCI structure with lookahead logic)
        while(done<np){
            int rem=np-done,k=std::max(1,(int)std::ceil(alpha*rem));
            std::vector<int>cands;for(int p:sorted)if(!al[p]){cands.push_back(p);if((int)cands.size()==k)break;}
            auto b=findBestInsertion(cands,sq,ld,d);
            if(b.found){
                sq[b.m].insert(sq[b.m].begin()+b.pos,b.p);ld[b.m]+=b.time;al[b.p]=true;done++;
                if(h==2){
                    bool imp=true;
                    while(imp){imp=false;
                        double cb=0;int pv=d.initial_state[b.m];
                        for(int p:sq[b.m]){cb+=d.setup_costs[IX(pv,p,b.m,np,nm)];pv=p;}
                        for(int i=0;i<(int)sq[b.m].size()-1;i++){
                            std::swap(sq[b.m][i],sq[b.m][i+1]);
                            double nl=0; pv=d.initial_state[b.m];
                            for(int p:sq[b.m]){nl+=d.setup_times[IX(pv,p,b.m,np,nm)]+d.demands[p]/d.production_rates[p*nm+b.m];pv=p;}
                            if(nl<=d.machine_capacities[b.m]){
                                double nc=0; pv=d.initial_state[b.m];
                                for(int p:sq[b.m]){nc+=d.setup_costs[IX(pv,p,b.m,np,nm)];pv=p;}
                                if(nc<cb){cb=nc;ld[b.m]=nl;imp=true;}
                                else std::swap(sq[b.m][i],sq[b.m][i+1]);}
                            else std::swap(sq[b.m][i],sq[b.m][i+1]);
                        }
                    }
                }
            } else{al[cands[0]]=true;done++;}
        }
    }
    else if(h==1){ // Balanced
        double lambda=alpha; double mcap=*std::max_element(d.machine_capacities.begin(),d.machine_capacities.end());
        while(done<np){
            int rem=np-done, k=std::max(1,(int)std::ceil(alpha*rem));
            std::vector<int>cands;for(int p:sorted)if(!al[p]){cands.push_back(p);if((int)cands.size()==k)break;}
            int bp=-1,bm=-1,bpos=-1;double bsc=1e18,bt=0;
            for(int p:cands)for(int m=0;m<nm;m++){
                if(d.production_rates[p*nm+m]<=1e-6)continue;
                double nl=ld[m]/mcap;
                for(int pos=0;pos<=(int)sq[m].size();pos++){
                    auto info=calcDelta(p,m,pos,sq,ld,d);
                    if(ld[m]+info.dt<=d.machine_capacities[m]){
                        double sc=lambda*info.dc+(1.0-lambda)*nl*100.0;
                        if(sc<bsc){bsc=sc;bt=info.dt;bp=p;bm=m;bpos=pos;}
                    }
                }
            }
            if(bp!=-1){sq[bm].insert(sq[bm].begin()+bpos,bp);ld[bm]+=bt;al[bp]=true;done++;}
            else{al[cands[0]]=true;done++;}
        }
    }
    else if(h==3){ // FixOpt
        int batch_k=2+(int)(alpha*3);if(batch_k>4)batch_k=4;if(batch_k<2)batch_k=2;
        while(done<np){
            int rem=np-done;int bk=std::min(batch_k,rem);
            std::vector<int>batch;for(int p:sorted)if(!al[p]){batch.push_back(p);if((int)batch.size()==bk)break;}
            if(bk<=1){
                auto b=findBestInsertion(batch,sq,ld,d);
                if(b.found){sq[b.m].insert(sq[b.m].begin()+b.pos,b.p);ld[b.m]+=b.time;}
                al[batch[0]]=true;done++;continue;
            }
            std::vector<int>perm=batch;std::sort(perm.begin(),perm.end());
            double bestC=1e18;std::vector<std::vector<int>>bestSq;std::vector<double>bestLd;
            do{
                auto tSq=sq;auto tLd=ld;double tC=0;bool ok=true;
                for(int p:perm){
                    auto b=findBestInsertion({p},tSq,tLd,d);
                    if(b.found){tSq[b.m].insert(tSq[b.m].begin()+b.pos,b.p);tLd[b.m]+=b.time;tC+=b.cost;}
                    else{ok=false;tC+=1e9;break;}
                }
                if(tC<bestC){bestC=tC;bestSq=tSq;bestLd=tLd;}
            }while(std::next_permutation(perm.begin(),perm.end()));
            sq=bestSq;ld=bestLd;
            for(int p:batch){al[p]=true;done++;}
        }
    }
    else if(h==4){ // ATCS
        const double eps=0.01;
        while(done<np){
            int rem=np-done,k=std::max(1,(int)std::ceil(alpha*rem));
            std::vector<int>cands;for(int p:sorted)if(!al[p]){cands.push_back(p);if((int)cands.size()==k)break;}
            int bp=-1,bm=-1;double bsc=-1e18,bt=0;
            for(int p:cands){
                double urgency=1.0;for(int i=0;i<(int)sorted.size();i++)if(sorted[i]==p){urgency=1.0-(double)i/np;break;}
                for(int m=0;m<nm;m++){
                    double r=d.production_rates[p*nm+m];if(r<=1e-6)continue;
                    double pt=d.demands[p]/r;
                    int last=sq[m].empty()?d.initial_state[m]:sq[m].back();
                    double sc=d.setup_costs[IX(last,p,m,np,nm)]; double st=d.setup_times[IX(last,p,m,np,nm)];
                    if(ld[m]+st+pt<=d.machine_capacities[m]){
                        double score=(d.demands[p]/(sc+eps))*urgency;
                        if(score>bsc){bsc=score;bt=st+pt;bp=p;bm=m;}
                    }
                }
            }
            if(bp!=-1){sq[bm].push_back(bp);ld[bm]+=bt;al[bp]=true;done++;}
            else{al[cands[0]]=true;done++;}
        }
    }

    printf("=== SEQUENCE_START ===\n[");
    for(int m=0;m<nm;m++){printf("[");
        for(size_t i=0;i<sq[m].size();i++){printf("%d",sq[m][i]);if(i<sq[m].size()-1)printf(", ");}
        printf("]");if(m<nm-1)printf(", ");}
    printf("]\n=== SEQUENCE_END ===\n");
}

#endif
