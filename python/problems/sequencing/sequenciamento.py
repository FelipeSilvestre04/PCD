import numpy as np
import os
import time
import sys

# Adiciona o caminho para importar RKO e Environment
current_directory = os.path.dirname(os.path.abspath(__file__))
python_directory = os.path.dirname(os.path.dirname(current_directory))
sys.path.insert(0, python_directory)

from rko.rko import RKO
from rko.environment import RKOEnvAbstract

class Sequenciamento(RKOEnvAbstract):

    def __init__(self, instance_path: str):
        self.save_q_learning_report = False
        self.max_time = 150
        
        # Leitura da instÃ¢ncia
        instances_dir = os.path.abspath(os.path.join(current_directory, '..', '..', '..', 'instances'))
        self.tempos_maquinas, self.custo_maquinas = self.read_instance(os.path.join(instances_dir, instance_path))
        
        # Ajuste de atributos para clareza
        self.itens = self.num_products
        self.maquinas = self.num_machines

        self.instance_name = instance_path.split('/')[-1]
        self.LS_type: str = 'Best'
        self.dict_best: dict = {}

        # --- MUDANÃ‡A PRINCIPAL: Tamanho do Vetor ---
        # Produtos (N) + Separadores (M - 1)
        # Ex: 5 produtos, 3 mÃ¡quinas -> precisamos de 2 cortes para criar 3 grupos.
        self.tam_solution = self.num_products 
        
        # --- ParÃ¢metros (Mantidos) ---
        self.BRKGA_parameters = {
            'p': [1000, 500], 'pe': [0.20, 0.15], 'pm': [0.05], 'rhoe': [0.70]
        }
        self.SA_parameters = {
            'SAmax': [10, 5], 'alphaSA': [0.5, 0.7], 'betaMin': [0.01, 0.03], 
            'betaMax': [0.05, 0.1], 'T0': [1000]
        }
        self.ILS_parameters = {
            'betaMin': [0.10,0.5], 'betaMax': [0.20,0.15]
        }
        self.VNS_parameters = {
            'kMax': [5,3], 'betaMin': [0.05, 0.1]
        }
        self.PSO_parameters = {
            'PSize': [1000,500], 'c1': [2.05], 'c2': [2.05], 'w': [0.73]
        }
        self.GA_parameters = {
            'sizePop': [1000,500], 'probCros': [0.98], 'probMut': [0.005, 0.01]
        }
        self.LNS_parameters = {
            'betaMin': [0.10], 'betaMax': [0.30], 'TO': [10000], 'alphaLNS': [0.95,0.9]
        }

    def read_instance(self, instance_path):
        """
        LÃª a instÃ¢ncia do arquivo no formato Li & Milne.
        """
        if not os.path.exists(instance_path):
            raise FileNotFoundError(f"Arquivo nÃ£o encontrado: {instance_path}")

        try:
            with open(instance_path, 'r') as f:
                tokens = f.read().split()
            
            iterator = iter(tokens)

            # 1. DimensÃµes BÃ¡sicas
            self.num_products = int(next(iterator))
            self.num_machines = int(next(iterator))

            # 2. Capacidades das MÃ¡quinas (T_r)
            self.machine_capacities = np.array([float(next(iterator)) for _ in range(self.num_machines)])

            # 3. Produtos Iniciais nas MÃ¡quinas (P^0_r)
            self.initial_state = [int(next(iterator)) for _ in range(self.num_machines)]

            # 4. Demandas dos Produtos (d_i)
            self.demands = np.array([float(next(iterator)) for _ in range(self.num_products)])

            # 5. Taxas de ProduÃ§Ã£o (p_ir)
            self.production_rates = np.zeros((self.num_products, self.num_machines))
            for i in range(self.num_products):
                for r in range(self.num_machines):
                    self.production_rates[i][r] = float(next(iterator))

            # 6. Custos de Setup (c_ijr)
            self.setup_costs = np.zeros((self.num_products, self.num_products, self.num_machines))
            for r in range(self.num_machines):
                label = next(iterator) 
                if "Reactor_Cost" not in label:
                    raise ValueError(f"Esperado label de custo, encontrado: {label}")
                
                for i in range(self.num_products):
                    for j in range(self.num_products):
                        self.setup_costs[i][j][r] = float(next(iterator))

            # 7. Tempos de Setup (t_ijr)
            self.setup_times = np.zeros((self.num_products, self.num_products, self.num_machines))
            for r in range(self.num_machines):
                label = next(iterator)
                if "Reactor_Time" not in label:
                    raise ValueError(f"Esperado label de tempo, encontrado: {label}")

                for i in range(self.num_products):
                    for j in range(self.num_products):
                        self.setup_times[i][j][r] = float(next(iterator))

            print(f"InstÃ¢ncia carregada: {self.num_products} produtos, {self.num_machines} mÃ¡quinas.")
            return self.setup_times, self.setup_costs

        except StopIteration:
            raise ValueError("O arquivo de instÃ¢ncia terminou inesperadamente.")
        except Exception as e:
            raise ValueError(f"Erro ao ler o arquivo: {str(e)}")

    def decoder(self, keys: np.ndarray) -> np.ndarray:
        """
        Decodificador: transforma random keys em uma permutaÃ§Ã£o de produtos.
        
        Args:
            keys: vetor de random keys (floats entre 0 e 1)
            
        Returns:
            np.ndarray: permutaÃ§Ã£o dos produtos (ordem de prioridade)
        """
        return np.argsort(keys)

    def cost(self, solution: np.ndarray, final_solution: bool = False) -> float:
        """
        Calcula o custo de uma soluÃ§Ã£o (permutaÃ§Ã£o de produtos).
        
        Usa a heurÃ­stica de InserÃ§Ã£o Mais Barata (Best Fit) para alocar
        os produtos nas mÃ¡quinas na ordem dada pela permutaÃ§Ã£o.
        
        Args:
            solution: permutaÃ§Ã£o dos produtos (array de Ã­ndices)
            final_solution: se True, imprime detalhes da soluÃ§Ã£o
            
        Returns:
            float: custo total (setup + penalidades)
        """
        sorted_products = solution
        
        # Estruturas da soluÃ§Ã£o
        machine_sequences = [[] for _ in range(self.num_machines)]
        machine_loads = np.zeros(self.num_machines)
        
        total_setup_cost = 0.0
        penalty = 0.0
        
        # Para cada produto, busca o MELHOR lugar (menor custo)
        for product_idx in sorted_products:
            best_machine = -1
            best_position = -1
            min_delta_cost = float('inf')
            best_time_increase = 0.0
            
            # Itera sobre todas as mÃ¡quinas
            for m in range(self.num_machines):
                # Verifica se a mÃ¡quina consegue processar o produto
                rate = self.production_rates[product_idx][m]
                if rate <= 1e-6: continue 
                
                prod_time = self.demands[product_idx] / rate
                current_seq = machine_sequences[m]
                
                # Testa TODAS as posiÃ§Ãµes de inserÃ§Ã£o
                for pos in range(len(current_seq) + 1):
                    if pos == 0:
                        prev_prod = self.initial_state[m]
                    else:
                        prev_prod = current_seq[pos-1]
                    
                    if pos < len(current_seq):
                        next_prod = current_seq[pos]
                    else:
                        next_prod = None
                    
                    # --- Delta Custo ---
                    cost_add = self.setup_costs[prev_prod][product_idx][m]
                    cost_rem = 0.0
                    
                    if next_prod is not None:
                        cost_add += self.setup_costs[product_idx][next_prod][m]
                        cost_rem = self.setup_costs[prev_prod][next_prod][m]
                    
                    delta_cost = cost_add - cost_rem
                    
                    # --- Delta Tempo ---
                    time_add = self.setup_times[prev_prod][product_idx][m]
                    time_rem = 0.0
                    
                    if next_prod is not None:
                        time_add += self.setup_times[product_idx][next_prod][m]
                        time_rem = self.setup_times[prev_prod][next_prod][m]
                        
                    delta_time = time_add + prod_time - time_rem
                    
                    # Verifica Capacidade
                    if machine_loads[m] + delta_time <= self.machine_capacities[m]:
                        if delta_cost < min_delta_cost:
                            min_delta_cost = delta_cost
                            best_machine = m
                            best_position = pos
                            best_time_increase = delta_time
                        elif abs(delta_cost - min_delta_cost) < 1e-6:
                            if best_machine != -1:
                                current_rate = self.production_rates[product_idx][best_machine]
                                if rate > current_rate:
                                    best_machine = m
                                    best_position = pos
                                    best_time_increase = delta_time

            # Realiza a alocaÃ§Ã£o na melhor opÃ§Ã£o encontrada
            if best_machine != -1:
                machine_sequences[best_machine].insert(best_position, product_idx)
                machine_loads[best_machine] += best_time_increase
                total_setup_cost += min_delta_cost
            else:
                # Inviabilidade: nÃ£o coube em lugar nenhum
                penalty += 100000 + (self.demands[product_idx] * 1000)

        total_cost = total_setup_cost + penalty
        
        if final_solution:
            print("\n" + "="*40)
            print(f"SOLUÃ‡ÃƒO FINAL - {self.instance_name}")
            print("="*40)
            print(f"Objetivo Total: {total_cost:.2f}")
            print(f"Custo de Setup: {total_setup_cost:.2f}")
            print(f"Penalidade: {penalty:.2f}")
            print("-" * 20)
            
            for m, seq in enumerate(machine_sequences):
                load = machine_loads[m]
                cap = self.machine_capacities[m]
                p0 = self.initial_state[m]
                utilizacao = (load / cap) * 100 if cap > 0 else 0
                status = "OK" if load <= cap else "ESTOUROU"
                
                print(f"MÃ¡q {m} [InÃ­cio {p0}]: {len(seq)} itens | Carga: {load:.1f}/{cap:.1f} ({utilizacao:.1f}%) [{status}]")
            print("="*40 + "\n")
                
        return total_cost

if __name__ == "__main__":
    for num in range(100, 701, 100):
        for reactors in [5, 10, 15]:
    #         # if (num == 600 and reactors == 5) or (num == 600 and reactors == 10):
    #         #     continue
            env = Sequenciamento(f'{num}_{reactors}_v2.txt')
    #         start_time = time.time()
    #         # solutions = 0
    #         # while time.time() - start_time < 60:
    #         #     random_solution = np.random.rand(env.tam_solution)
    #         #     decoded = env.decoder(random_solution)
    #         #     cost = env.cost(decoded)
    #         #     solutions += 1
    #         # print(f'InstÃ¢ncia: {num}_{reactors}_v2.txt | SoluÃ§Ãµes em 60s: {solutions}')
            results_dir = os.path.join(current_directory, 'Results')
            solver = RKO(env, True, save_directory=os.path.join(results_dir, 'RKO_testes.txt'))
            final_cost, final_solution, time_to_best = solver.solve(time_total=150, brkga=1, lns=1, vns=1, ils=1, sa=1, pso=1, ga=1, ms=1, runs=10)
    #         env.cost(env.decoder(final_solution), final_solution=True)

    # for num in range(100, 701, 100):
    #     for reactors in [5, 10, 15]:

    #         env = Sequenciamento(f'{num}_{reactors}_v2.txt')
    #         start_time = time.time()
    #         solutions = 0
    #         while time.time() - start_time < 60:
    #             random_solution = np.random.rand(env.tam_solution)
    #             decoded = env.decoder(random_solution)
    #             cost = env.cost(decoded)
    #             solutions += 1
            
    #         with open(os.path.join(current_directory, 'Results', 'RKO_testes.txt'), "a") as f:
    #             f.write(f'Instancia: {num}_{reactors}_v2.txt | Solucoes em 60s: {solutions}\n')