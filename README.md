# PCD — Python e C++: RKO/BRKGA com Best Fit

Experimento da disciplina de Computação Concorrente e Distribuída para o problema MM-SDCO.

## Conteúdo

- `campaign.py`: execução sequencial das configurações, instrumentação e reconstrução dos custos.
- `python/`: implementação Python utilizada (processos e pool via Manager).
- `cpp/src/` e `cpp/config/`: fontes C++ instrumentadas e parâmetros utilizados (OpenMP). A licença e a autoria dessas fontes estão em `cpp/LICENSE` e `cpp/Authors`.
- `instances/`: dez instâncias, com 100/200/300/400/500 produtos e 5/10 máquinas.
- `results_120s/`: dados finais das 500 execuções; cada diretório contém a solução e as métricas individuais em `result.json`.
- `analyze.py`: médias, desvios padrão, IC de 95%, speedup de vazão, Wilcoxon pareado e correção de Holm.

## Resultados

Foram realizadas cinco repetições de 120 segundos por combinação de linguagem, instância e 1/2/4/6/8 trabalhadores: 100 configurações e 500 execuções, todas viáveis e com custos reconstruídos conferidos. As sementes são distintas entre as repetições de cada configuração.

Arquivos principais em `results_120s/`:

- `results.csv`: uma linha por execução.
- `per_configuration_statistics.csv`: média, desvio padrão e IC por configuração.
- `scalability.csv`: speedup e eficiência paralela calculados com avaliações por segundo.
- `paired_vs_cpp1.csv`: Python em cada nível versus C++ com um trabalhador.
- `exploratory_tests.csv`: testes pareados sobre as médias das dez instâncias.
- `RELATORIO.md`: resumo da análise.
- `metadata.json`: ambiente e parâmetros registrados durante a campanha.

## Reproduzir

Usar Linux ou Ubuntu/WSL, Python 3.12 e G++ com suporte a OpenMP. O ambiente da campanha foi Ubuntu via WSL1, CPython 3.12.3, G++ 13.3.0 e Intel i7-9700K (8 núcleos), com aproximadamente 16 GB de RAM.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash build.sh
python analyze.py results_120s
python verify_results.py
```

Para uma nova campanha (aproximadamente 16h40 de busca, além da inicialização):

```bash
PYTHON=python bash run.sh
```

Por padrão, novas execuções ficam em `results_new/`, preservando os dados históricos. A retomada ignora resultados já salvos nesse diretório. Executar apenas uma campanha por vez. Para uma verificação curta de execução:

```bash
python campaign.py --smoke --seconds 2 --repetitions 1 --output smoke
```

## Interpretação

População 1000, elite 0,20, parâmetro de mutação 0,05, herança 0,70 e pool 20. O decoder e seus desempates foram alinhados, mas os operadores BRKGA, a busca local e os reinícios diferem entre as implementações. Portanto, a comparação não isola o efeito da linguagem.

As avaliações são chamadas concluídas do decoder e podem repetir candidatos. O tempo registrado pode ultrapassar os 120 segundos e a inicialização é registrada separadamente. O speedup de vazão usa avaliações/s em relação a um trabalhador da mesma linguagem e instância. Os testes usam dez observações pareadas (médias por instância), não 50 execuções como instâncias independentes.

Energia elétrica não foi medida. A estimativa discutida no trabalho usa `E = (95/8) × trabalhadores × segundos / 3600`, em Wh, com utilização integral assumida. É um modelo de CPU baseado no TDP, sem RAM, consumo de base ou processos auxiliares.

Os dados foram copiados sem alteração da campanha em `PO_Ball/experiments/brkga_python_cpp`. As adaptações neste repositório alteram somente caminhos para tornar a execução independente do repositório original. O artigo LaTeX não faz parte deste repositório.
