"""Sequential campaign: existing Python BRKGA first, then existing C++ BRKGA."""
import argparse
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON = HERE / 'python'
sys.path.insert(0, str(PYTHON))
sys.path.insert(0, str(PYTHON / 'problems/sequencing'))
from sequenciamento import Sequenciamento
from rko.rko import RKO, SolutionPool

class Deadline(Exception):
    pass

class MeasuredEnv(Sequenciamento):
    def decoder(self, keys):
        return np.argsort(keys, kind='stable')

    def cost(self, solution, final_solution=False):
        if hasattr(self, 'deadline') and time.perf_counter() >= self.deadline:
            raise Deadline()
        cost = super().cost(solution, final_solution)
        if hasattr(self, 'evaluations'):
            self.evaluations += 1
            if cost < self.measured_best:
                self.measured_best = float(cost)
                self.measured_permutation = list(map(int, solution))
                self.measured_time = time.perf_counter() - self.measured_start
                self.history.append([self.measured_time, float(cost)])
        return cost

def worker(env, pool, tag, seed, ready, go, start, seconds, directory):
    result = {'worker': tag, 'status': 'ok'}
    try:
        random.seed(seed + 104729 * (tag + 1))
        np.random.seed((seed + 104729 * (tag + 1)) % 2**32)
        ready.release()
        if not go.wait(90):
            raise RuntimeError('Start signal timed out')
        env.measured_start = start.value
        env.deadline = start.value + seconds
        env.evaluations = 0
        env.measured_best = float('inf')
        env.measured_permutation = None
        env.measured_time = None
        env.history = []
        runner = RKO(env, print_best=False)
        runner.max_time = max(0.001, env.deadline-time.perf_counter())
        runner.start_time = time.time()
        try:
            runner.BRKGA(tag, pool)
        except Deadline:
            pass
        result.update(evaluations=env.evaluations, best_cost=env.measured_best,
                      permutation=env.measured_permutation,
                      time_to_best_s=env.measured_time, history=env.history)
    except Exception:
        result.update(status='error', error=traceback.format_exc())
    finally:
        (Path(directory) / f'worker_{tag}.json').write_text(json.dumps(result))

def python_run(instance, workers, seconds, seed, directory):
    init_start = time.perf_counter()
    env = MeasuredEnv(instance)
    env.max_time = seconds
    env.BRKGA_parameters = {'p': [1000], 'pe': [.2], 'pm': [.05], 'rhoe': [.7]}
    random.seed(seed); np.random.seed(seed)
    with mp.Manager() as manager:
        pair = manager.list([float('inf'), None, None])
        pool = SolutionPool(20, manager.list(), pair, lock=manager.Lock(), env=env)
        for _ in range(20):
            keys = np.random.random(env.tam_solution)
            pool.insert((env.cost(env.decoder(keys)), list(keys)), 'initial', -1)
        initial = float(pair[0]); initial_keys = list(pair[1])
        ready = mp.Semaphore(0); go = mp.Event(); start = mp.Value('d', 0)
        processes = [mp.Process(target=worker, args=(env,pool,i,seed,ready,go,start,seconds,str(directory)))
                     for i in range(workers)]
        for p in processes:
            p.start()
        for _ in processes:
            if not ready.acquire(timeout=90):
                for p in processes:
                    p.terminate(); p.join()
                raise RuntimeError('Worker initialization failed')
        initialization = time.perf_counter()-init_start
        start.value = time.perf_counter(); go.set()
        for p in processes:
            p.join(timeout=max(0, start.value+seconds+60-time.perf_counter()))
        elapsed = time.perf_counter()-start.value
        for p in processes:
            if p.is_alive():
                p.terminate(); p.join()
                raise RuntimeError('Worker exceeded deadline grace period')
            if p.exitcode != 0:
                raise RuntimeError(f'Worker exit code {p.exitcode}')
        details = [json.loads((directory/f'worker_{i}.json').read_text()) for i in range(workers)]
        for item in details:
            if item['status'] != 'ok':
                raise RuntimeError(item.get('error'))
        best = {'best_cost': initial, 'time_to_best_s': 0, 'keys': initial_keys}
        history = [[0, initial]]
        for item in details:
            history.extend(item['history'])
            if item['best_cost'] < best['best_cost']:
                keys = np.empty(env.tam_solution)
                keys[item['permutation']] = np.arange(env.tam_solution)/env.tam_solution
                best = {'best_cost': item['best_cost'], 'time_to_best_s': item['time_to_best_s'], 'keys': keys.tolist()}
        improvements=[]; current=float('inf')
        for t,c in sorted(history):
            if c < current:
                improvements.append([t,c]); current=c
        evaluations = sum(item['evaluations'] for item in details)
        return dict(language='python', workers=workers, budget_s=seconds, elapsed_s=elapsed,
                    initialization_s=initialization, evaluations=evaluations,
                    evals_per_s=evaluations/elapsed, history=improvements, **best)

def reconstruct(env, keys):
    """Independent reconstruction: feasibility, penalty, loads and sequences."""
    sequences = [[] for _ in range(env.num_machines)]
    loads = [0.] * env.num_machines; setup=0.; penalty=0.; omitted=[]
    for p in np.argsort(keys, kind='stable'):
        best=None; min_cost=float('inf')
        for m, seq in enumerate(sequences):
            rate=env.production_rates[p,m]
            if rate<=1e-6: continue
            for pos in range(len(seq)+1):
                prev=env.initial_state[m] if pos==0 else seq[pos-1]
                nxt=seq[pos] if pos<len(seq) else None
                dc=env.setup_costs[prev,p,m]; dt=env.setup_times[prev,p,m]+env.demands[p]/rate
                if nxt is not None:
                    dc+=env.setup_costs[p,nxt,m]-env.setup_costs[prev,nxt,m]
                    dt+=env.setup_times[p,nxt,m]-env.setup_times[prev,nxt,m]
                if loads[m]+dt<=env.machine_capacities[m]:
                    if dc<min_cost:
                        min_cost=dc; best=(m,pos,dt)
                    elif abs(dc-min_cost)<1e-6 and best is not None and rate>env.production_rates[p,best[0]]:
                        best=(m,pos,dt)
        if best is None:
            omitted.append(int(p)); penalty+=100000+env.demands[p]*1000
        else:
            m,pos,dt=best; sequences[m].insert(pos,int(p)); loads[m]+=dt; setup+=min_cost
    return dict(feasible=not omitted, unallocated=omitted, penalty=float(penalty),
                setup_cost=float(setup), sequences=sequences, loads=loads, reconstructed_cost=float(setup+penalty))

def validate(output):
    env=MeasuredEnv('100_5_v2.txt')
    keys_path=output/'validation_keys.txt'
    for i in range(8):
        keys=np.random.default_rng(1234+i).random(env.tam_solution) if i<6 else np.zeros(env.tam_solution)
        if i==7: keys=np.arange(env.tam_solution)[::-1]/env.tam_solution
        np.savetxt(keys_path,keys,fmt='%.17g')
        value=float(subprocess.check_output([str(HERE/'cpp/benchmark'),
                    str(HERE/'instances/100_5_v2.txt'),'1','0','1234',str(keys_path)],cwd=HERE/'cpp',text=True))
        expected=env.cost(env.decoder(keys))
        assert np.isclose(value,expected,rtol=1e-12,atol=1e-6),(value,expected)
        assert np.isclose(value,reconstruct(env,keys)['reconstructed_cost'],rtol=1e-12,atol=1e-6)
    (output/'validation.json').write_text(json.dumps({'cases':8,'decoder_equivalence':'passed'}))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seconds',type=int,default=120)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--repetitions',type=int,default=1)
    parser.add_argument('--first-repeat',type=int,default=1)
    parser.add_argument('--output',default='results_120s')
    args=parser.parse_args()
    if not 1 <= args.first_repeat <= args.repetitions:
        parser.error('Require 1 <= first-repeat <= repetitions')
    output=(HERE/args.output).resolve(); output.mkdir(parents=True,exist_ok=True)
    instances=[f'{n}_{m}_v2.txt' for n in [100,200,300,400,500] for m in [5,10]]
    levels=[1,2,4,6,8]
    if args.smoke: instances=[instances[0],instances[-1]]; levels=[1,8]
    for instance in instances:
        assert (HERE/'instances'/instance).is_file(),instance
    validate(output)
    metadata=dict(platform=platform.platform(),python=sys.version,numpy=np.__version__,
                  instances=instances,levels=levels,budget_s=args.seconds,repetitions=args.repetitions,
                  seed=20261007,energy='not collected',params={'p':1000,'pe':.2,'pm':.05,'rhoe':.7},
                  differences=['Python mutation: per gene 0.05; C++ random mutant fraction and local search',
                               'Python restarts population halfway and uses shared pool as parent; C++ differs',
                               'Measurement compares existing implementations, not isolated language cost'])
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    fields=['language','instance','workers','budget_s','elapsed_s','initialization_s','evaluations',
            'evals_per_s','best_cost','time_to_best_s','feasible','penalty','setup_cost','seed','repetition']
    tasks=[(language,repetition,instance,workers)
           for language in ['python','cpp']
           for repetition in range(args.first_repeat,args.repetitions+1)
           for instance in instances for workers in levels]
    for language,repetition,instance,workers in tasks:
        suffix='' if repetition==1 else f'_r{repetition}'
        directory=output/f'{language}_{instance[:-4]}_p{workers}{suffix}'
        directory.mkdir(exist_ok=True)
        result_path=directory/'result.json'
        if result_path.exists(): continue
        seed=20261007+int(instance.split('_')[0])*100+int(instance.split('_')[1])+(repetition-1)*10000000
        print(f'{language} {instance} workers={workers} repetition={repetition}/{args.repetitions} seconds={args.seconds}',flush=True)
        if language=='python':
            result=python_run(instance,workers,args.seconds,seed,directory)
        else:
            subprocess.run([str(HERE/'cpp/benchmark'),str(HERE/'instances'/instance),
                            str(workers),str(args.seconds),str(seed),str(directory/'raw_cpp.json')],
                           cwd=HERE/'cpp',check=True,timeout=args.seconds+120)
            result=json.loads((directory/'raw_cpp.json').read_text())
        result.update(instance=instance,seed=seed,repetition=repetition)
        env=MeasuredEnv(instance)
        result.update(reconstruct(env,result['keys']))
        if not np.isclose(result['best_cost'],result['reconstructed_cost'],rtol=1e-12,atol=1e-6):
            raise RuntimeError('Saved solution failed cost validation')
        result_path.write_text(json.dumps(result,indent=2))
        all_results=[json.loads(p.read_text()) for p in sorted(output.glob('*/result.json'))]
        for item in all_results: item.setdefault('repetition',1)
        with (output/'results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(all_results)
        print(f"completed: {result['evaluations']} evals; cost={result['best_cost']:.6f}; elapsed={result['elapsed_s']:.3f}",flush=True)
        subprocess.run([sys.executable,str(HERE/'analyze.py'),str(output)],check=True)
    print('CAMPAIGN COMPLETE',flush=True)

if __name__=='__main__':
    mp.set_start_method('spawn')
    main()
