#!/bin/python3

import subprocess
import re
import pandas as pd

num_clients = 100

comm_rows = []
time_rows = []

for strategy in ["Bit-Splitting", "Sorting"]:
    for num_clients in [1, 10, 50, 100, 1000]:
        for i in range(0, 100001, 10000):
            if i == 0:
                i = 1

            print(i);
            args = ["cargo", "run", "--release", "--example", "optimization"]
            if strategy == "Sorting":
                args += ["--features", "ram"]
            args += [str(num_clients), str(i), "1024", "opt"]
            p = subprocess.run(args, capture_output=True)
            client_server = re.search(b"Client -> Server opt: (\d*)", p.stdout)
            if client_server is None:
                print("Error! got:", p.stdout, "for", args)
                exit(1)
            server_server = re.search(b"Server -> Server opt: (\d*)", p.stdout)
            if server_server is None:
                print("Error! got:", p.stdout, "for", args)
                exit(1)
            server_client = re.search(b"Server -> Client opt: (\d*)", p.stdout)
            if server_client is None:
                print("Error! got:", p.stdout, "for", args)
                exit(1)


            client_time = re.search(b"Client time: (\d*)", p.stdout)
            if client_time is None:
                print("Error! got:", p.stdout, "for", args)
                exit(1)
            verif_time = re.search(b"Verify time: (\d*)", p.stdout)
            if verif_time is None:
                print("Error! got:", p.stdout, "for", args)
                exit(1)
            aggregate_time = re.search(b"Aggregate time: (\d*)", p.stdout)
            if aggregate_time is None:
                print("Error! got:", p.stdout, "for", args)
                exit(1)
            client_opt_time = re.search(b"Client Opt time: (\d*)", p.stdout)
            if client_opt_time is None:
                print("Error! got:", p.stdout, "for", args)
                exit(1)
            server_opt_time = re.search(b"Server Opt time: (\d*)", p.stdout)
            if server_opt_time is None:
                print("Error! got:", p.stdout, "for", args)
                exit(1)
            # print(client_server.group(1))
            # print(server_server.group(1))
            # print(server_client.group(1))
            comm_rows.append([strategy, i, num_clients, int(client_server.group(1)), int(server_server.group(1)), int(server_client.group(1))])
            time_rows.append([strategy, i, num_clients, int(client_time.group(1)), int(verif_time.group(1)), int(aggregate_time.group(1)), int(client_opt_time.group(1)), int(server_opt_time.group(1))])

df = pd.DataFrame(comm_rows, columns=["Proof_Strategy", "Schedule_Size", "Num_Clients", "Client_to_Server", "Server_to_Server", "Server_to_Client"])
df.to_csv("communication_opt.csv")

df = pd.DataFrame(time_rows, columns=["Proof_Strategy", "Schedule_Size", "Num_Clients", "Client_Nanos", "Verify_Nanos", "Aggregate_Nanos", "Client_Opt_Nanos", "Server_Ops_Nanos"])
df.to_csv("times_opt.csv")
print(df)
