#!/bin/python3

import subprocess
import re
import pandas as pd

num_clients = 1

comm_rows = []
time_rows = []

for strategy in ["Bit-Splitting", "Sorting"]:
    for i in range(0, 100001, 10000):
        if i == 0:
            i = 1

        print(i);
        args = ["cargo", "run", "--release", "--example", "optimization"]
        if strategy == "Sorting":
            args += ["--features", "ram"]
        args += [str(num_clients), str(i), "1024"]
        p = subprocess.run(args, capture_output=True)
        client_server = re.search(b"Client -> Server: (\d*)", p.stdout)
        if client_server is None:
            print("Error! got:", p.stdout, "for", args)
            exit(1)
        server_server = re.search(b"Server -> Server: (\d*)", p.stdout)
        if server_server is None:
            print("Error! got:", p.stdout, "for", args)
            exit(1)
        server_client = re.search(b"Server -> Client: (\d*)", p.stdout)
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
        # print(client_server.group(1))
        # print(server_server.group(1))
        # print(server_client.group(1))
        comm_rows.append([strategy, i, int(client_server.group(1)), int(server_server.group(1)), int(server_client.group(1))])
        time_rows.append([strategy, i, int(client_time.group(1)), int(verif_time.group(1)), int(aggregate_time.group(1))])

df = pd.DataFrame(comm_rows, columns=["Proof_Strategy", "Schedule_Size", "Client_to_Server", "Server_to_Server", "Server_to_Client"])
df.to_csv("communication.csv")

df = pd.DataFrame(time_rows, columns=["Proof_Strategy", "Schedule_Size", "Client_Nanos", "Verify_Nanos", "Aggregate_Nanos"])
df.to_csv("times.csv")
print(df)
