# Análise da campanha BRKGA — Python e C++

Execuções concluídas: 500/500. Orçamento: 120 s por configuração.

Meta: 5 repetições por configuração, com sementes distintas. Resultados anteriores pertencem à repetição 1.
per_configuration_statistics.csv contém média, desvio padrão amostral, melhor custo observado e IC de 95% da média (Student t; ausente quando n=1).
Intervalos com cinco amostras têm pouca precisão e pressupõem que a distribuição das médias seja adequadamente aproximada pelo modelo t.
Tempos de inicialização são separados. A janela medida inclui comunicação e uma possível última avaliação que atravessa o limite.
Comparação das implementações existentes: apesar de decoder e parâmetros comuns, operadores, busca local e reinício diferem.
Energia não coletada: nenhum contador RAPL foi identificado no ambiente WSL.

## Estatísticas descritivas

```text
                  instances  median_evals_per_s  mean_evals_per_s  median_elapsed_s  feasible
language workers                                                                             
cpp      1               10         2126.681013       7007.969455        120.000720        10
         2               10         4130.792280      13619.464018        120.001827        10
         4               10         7866.481009      26269.535265        120.001996        10
         6               10        11263.182889      37940.143108        120.002355        10
         8               10        13551.580486      47530.549693        120.006752        10
python   1               10            9.825251         24.503536        120.163740        10
         2               10           19.092361         46.171843        120.134181        10
         4               10           35.494668         83.119698        120.238002        10
         6               10           48.682466        114.825930        120.309455        10
         8               10           58.879631        137.271845        122.756614        10
```

## Comparação com C++ de um trabalhador

Razão de vazão = avaliações/s Python ÷ avaliações/s C++ com 1 trabalhador. Critério operacional de proximidade: razão ≥ 0,90.
Gap de custo = 100 × (custo Python − custo C++) ÷ |custo C++|; menor é melhor. Penalidades e viabilidade são registradas separadamente.
O tempo até a melhor solução é descritivo: cada execução pode atingir um custo diferente. Não representa tempo para um alvo comum.

## Inferência exploratória

Wilcoxon pareado bilateral sobre log da razão de vazão média e gap relativo de custo médio (somente pares com todas as repetições viáveis), com correção de Holm.
A média das repetições de cada instância é uma observação pareada: as 50 execuções não são tratadas como 50 instâncias independentes.
Os testes aguardam as cinco repetições dos dois lados em todas as dez instâncias. O teste pressupõe simetria das diferenças; a seleção das instâncias limita generalização.
Fonte: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html

                    metric  python_workers  n  median_difference  p_value   p_holm
      log_throughput_ratio               1 10          -5.338005 0.001953 0.019531
relative_cost_gap_feasible               1 10           0.117632 0.001953 0.019531
      log_throughput_ratio               2 10          -4.671423 0.001953 0.019531
relative_cost_gap_feasible               2 10           0.095118 0.001953 0.019531
      log_throughput_ratio               4 10          -4.037554 0.001953 0.019531
relative_cost_gap_feasible               4 10           0.073632 0.001953 0.019531
      log_throughput_ratio               6 10          -3.669994 0.001953 0.019531
relative_cost_gap_feasible               6 10           0.069342 0.001953 0.019531
      log_throughput_ratio               8 10          -3.495129 0.001953 0.019531
relative_cost_gap_feasible               8 10           0.065850 0.001953 0.019531
