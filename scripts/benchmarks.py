
import time, subprocess, csv

with open("out_times_u16.csv", 'w') as out_file:
    writer = csv.writer(out_file)
    writer.writerow(["num_clients","num_slices","time_s"])
    for c in range(2, 100, 5):
        for n in range(1, 86400, 5000):
            start = time.time()
            proc = subprocess.run(["cargo", "run", "--example", "simple", "--release", "--", str(c), str(n)],stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            end = time.time()
            writer.writerow([c, n, end-start])
            print(c,n,end-start)
