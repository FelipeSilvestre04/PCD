"""Descriptive statistics and exploratory paired tests across instances."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    output=Path(sys.argv[1]); raw=pd.read_csv(output/'results.csv')
    metadata=json.loads((output/'metadata.json').read_text())
    repetitions=metadata.get('requested_repetitions',metadata.get('repetitions',1))
    expected=len(metadata['instances'])*len(metadata['levels'])*2*repetitions
    if 'repetition' not in raw: raw['repetition']=1
    raw['repetition']=raw.repetition.fillna(1).astype(int)
    if raw.duplicated(['language','instance','workers','repetition']).any():
        raise ValueError('Duplicate repetition records')
    df=raw.groupby(['language','instance','workers'],as_index=False).agg(
        repetitions=('repetition','count'),evals_per_s=('evals_per_s','mean'),
        std_evals_per_s=('evals_per_s','std'),best_cost=('best_cost','mean'),
        std_cost=('best_cost','std'),best_observed_cost=('best_cost','min'),
        elapsed_s=('elapsed_s','mean'),std_elapsed_s=('elapsed_s','std'),
        feasible_runs=('feasible','sum'))
    df['feasible']=df.feasible_runs==df.repetitions
    df['complete']=df.repetitions==repetitions
    for metric,std in [('evals_per_s','std_evals_per_s'),('best_cost','std_cost'),('elapsed_s','std_elapsed_s')]:
        half=student_t.ppf(.975,df.repetitions-1)*df[std]/np.sqrt(df.repetitions)
        df[f'{metric}_ci95_low']=df[metric]-half
        df[f'{metric}_ci95_high']=df[metric]+half
    df.to_csv(output/'per_configuration_statistics.csv',index=False)
    summary=df.groupby(['language','workers']).agg(instances=('instance','count'),
        median_evals_per_s=('evals_per_s','median'),mean_evals_per_s=('evals_per_s','mean'),
        median_elapsed_s=('elapsed_s','median'),feasible=('feasible','sum'))
    summary.to_csv(output/'summary.csv')
    own=df[df.workers==1][['language','instance','evals_per_s']].rename(columns={'evals_per_s':'baseline_evals_per_s'})
    scaling=df.merge(own,on=['language','instance'])
    scaling['throughput_speedup']=scaling.evals_per_s/scaling.baseline_evals_per_s
    scaling['parallel_efficiency']=scaling.throughput_speedup/scaling.workers
    scaling.to_csv(output/'scalability.csv',index=False)
    pairs=[]; tests=[]
    baseline=df[(df.language=='cpp')&(df.workers==1)].set_index('instance')
    for workers in metadata['levels']:
        py=df[(df.language=='python')&(df.workers==workers)].set_index('instance')
        common=py.index.intersection(baseline.index)
        if len(common)==0: continue
        for instance in common:
            a=py.loc[instance]; b=baseline.loc[instance]
            pairs.append(dict(instance=instance,python_workers=workers,
                python_repetitions=int(a.repetitions),cpp_repetitions=int(b.repetitions),
                complete=bool(a.complete and b.complete),
                throughput_ratio=a.evals_per_s/b.evals_per_s,
                cost_gap_percent=100*(a.best_cost-b.best_cost)/abs(b.best_cost) if b.best_cost!=0 else np.nan,
                python_feasible=bool(a.feasible),cpp_feasible=bool(b.feasible)))
        # Do not run a test before all ten matched observations exist.
        if len(common)!=len(metadata['instances']): continue
        if not py.loc[common].complete.all() or not baseline.loc[common].complete.all(): continue
        x=np.log(py.loc[common].evals_per_s.to_numpy()/baseline.loc[common].evals_per_s.to_numpy())
        valid=py.loc[common].feasible.to_numpy() & baseline.loc[common].feasible.to_numpy()
        denom=np.abs(baseline.loc[common].best_cost.to_numpy())
        valid &= denom>0
        cost=((py.loc[common].best_cost.to_numpy()-baseline.loc[common].best_cost.to_numpy())/np.where(denom>0,denom,1))[valid]
        for name,values in [('log_throughput_ratio',x),('relative_cost_gap_feasible',cost)]:
            n=len(values)
            if n<5: continue
            values=np.round(values,12)
            p=1.0 if np.all(values==0) else float(wilcoxon(values,alternative='two-sided',zero_method='wilcox',method='auto').pvalue)
            tests.append(dict(metric=name,python_workers=workers,n=n,median_difference=float(np.median(values)),p_value=p))
    if pairs:
        pair_df=pd.DataFrame(pairs);pair_df.to_csv(output/'paired_vs_cpp1.csv',index=False)
        match=pair_df.groupby('instance').apply(lambda g: g.loc[g.complete & (g.throughput_ratio>=.9),'python_workers'].min(),include_groups=False)
        match.rename('min_python_workers_at_least_90pct_cpp1').to_csv(output/'matching_workers.csv')
    if tests:
        order=np.argsort([t['p_value'] for t in tests]); running=0.
        for rank,idx in enumerate(order):
            running=max(running,min(1.,tests[idx]['p_value']*(len(tests)-rank)))
            tests[idx]['p_holm']=running
        pd.DataFrame(tests).to_csv(output/'exploratory_tests.csv',index=False)
    lines=['# Análise da campanha BRKGA — Python e C++','',
           f'Execuções concluídas: {len(raw)}/{expected}. Orçamento: {metadata["budget_s"]} s por configuração.',
           '', f'Meta: {repetitions} repetições por configuração, com sementes distintas. Resultados anteriores pertencem à repetição 1.',
           'per_configuration_statistics.csv contém média, desvio padrão amostral, melhor custo observado e IC de 95% da média (Student t; ausente quando n=1).',
           'Intervalos com cinco amostras têm pouca precisão e pressupõem que a distribuição das médias seja adequadamente aproximada pelo modelo t.',
           'Tempos de inicialização são separados. A janela medida inclui comunicação e uma possível última avaliação que atravessa o limite.',
           'Comparação das implementações existentes: apesar de decoder e parâmetros comuns, operadores, busca local e reinício diferem.',
           'Energia não coletada: nenhum contador RAPL foi identificado no ambiente WSL.', '',
           '## Estatísticas descritivas','','```text',summary.to_string(),'```','',
           '## Comparação com C++ de um trabalhador','',
           'Razão de vazão = avaliações/s Python ÷ avaliações/s C++ com 1 trabalhador. Critério operacional de proximidade: razão ≥ 0,90.',
           'Gap de custo = 100 × (custo Python − custo C++) ÷ |custo C++|; menor é melhor. Penalidades e viabilidade são registradas separadamente.',
           'O tempo até a melhor solução é descritivo: cada execução pode atingir um custo diferente. Não representa tempo para um alvo comum.', '',
           '## Inferência exploratória','',
           'Wilcoxon pareado bilateral sobre log da razão de vazão média e gap relativo de custo médio (somente pares com todas as repetições viáveis), com correção de Holm.',
           'A média das repetições de cada instância é uma observação pareada: as 50 execuções não são tratadas como 50 instâncias independentes.',
           'Os testes aguardam as cinco repetições dos dois lados em todas as dez instâncias. O teste pressupõe simetria das diferenças; a seleção das instâncias limita generalização.',
           'Fonte: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html','']
    if tests:
        lines.extend([pd.DataFrame(tests).to_string(index=False),''])
    else:
        lines.append('Testes aguardam pares completos entre Python e C++.')
    (output/'RELATORIO.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(8,5))
    for language,color in [('python','tab:blue'),('cpp','tab:orange')]:
        for instance,g in df[df.language==language].groupby('instance'):
            g=g.sort_values('workers'); ax.plot(g.workers,g.evals_per_s,color=color,alpha=.25)
        g=df[df.language==language].groupby('workers').evals_per_s.median()
        if len(g): ax.plot(g.index,g.values,'o-',color=color,label=language+' (mediana)')
    ax.set(xlabel='Trabalhadores BRKGA',ylabel='Avaliações por segundo',yscale='log',xticks=metadata['levels'])
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'throughput.png',dpi=160);plt.close(fig)

if __name__=='__main__': main()
