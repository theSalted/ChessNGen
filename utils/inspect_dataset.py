import pyarrow.parquet as pq
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="datasets/lichess_1k/transitions.parquet")
parser.add_argument("--rows", type=int, default=3)
args = parser.parse_args()

t = pq.read_table(args.path)

print("Schema:")
print(t.schema)
print()
print(f"Total rows: {t.num_rows}")
print()

for i in range(min(args.rows, t.num_rows)):
    print(f"Row {i}:")
    print(f"  game_id: {t['game_id'][i]}")
    print(f"  t: {t['t'][i]}")
    print(f"  in_paths: {t['in_paths'][i].as_py()}")
    print(f"  out_path: {t['out_path'][i]}")
    print()
