import time, subprocess, csv

with open("out_comm_u32.csv", 'w') as out_file:
    writer = csv.writer(out_file)
    writer.writerow(["num_slices","input_size","prep_size","out_size"])
    #for c in range(2, 100, 5):
    for n in range(1, 86400, 5000):
        #start = time.time()
        proc = subprocess.run(["cargo", "run", "--example", "comm_cost", "--release", "--", str(2), str(n)], capture_output=True)
        #print(proc.stdout)
        #end = time.time()
        print(proc.stdout.decode().strip().split(","))
        writer.writerow(proc.stdout.decode().strip().split(","))
        #print(c,n,end-start)
