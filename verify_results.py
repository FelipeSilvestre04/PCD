import json, math, collections, hashlib
from pathlib import Path
import campaign

def main():
 root=Path(__file__).resolve().parent
 manifest=json.loads((root/'SHA256SUMS.json').read_text())
 for name,digest in manifest.items():
  assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
 groups=collections.defaultdict(list)
 for file in (root/'results_120s').glob('*/result.json'):
  row=json.loads(file.read_text());row.setdefault('repetition',1)
  groups[(row['language'],row['instance'],row['workers'])].append(row)
 expected={(l,f'{n}_{m}_v2.txt',p) for l in ('python','cpp') for n in (100,200,300,400,500) for m in (5,10) for p in (1,2,4,6,8)}
 assert set(groups)==expected
 current_instance=None
 env=None
 for key,rows in sorted(groups.items(),key=lambda item:item[0][1]):
  assert len(rows)==5 and {r['repetition'] for r in rows}==set(range(1,6)) and len({r['seed'] for r in rows})==5,key
  if current_instance!=key[1]:
   env=campaign.MeasuredEnv(key[1])
   current_instance=key[1]
  for row in rows:
   reconstructed=campaign.reconstruct(env,row['keys'])
   assert row['budget_s']==120 and row['feasible'] and reconstructed['feasible'],key
   assert math.isclose(row['best_cost'],reconstructed['reconstructed_cost'],rel_tol=1e-12,abs_tol=1e-6),key
 print('500 execucoes verificadas: sementes, viabilidade, custos reconstruidos e hashes.')
 if (root/'cpp/benchmark').exists():
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:campaign.validate(Path(tmp))
  print('Equivalencia do decoder: 8 casos verificados.')

if __name__=='__main__':main()
