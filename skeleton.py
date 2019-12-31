from scapy.all import *
import pandas as pd
import numpy as np
import sys
import socket 
import os
import csv
dataflow = [] # contain the unique ip_src/dst,port src/dst
storage = [] # stores all of the packets collected
total_list = [] # list with features thats inserted into csv file
def fields_extraction(x):
    # print x.sprintf("{IP:%IP.src%,%IP.dst%,}"
    #     "{TCP:%TCP.sport%,%TCP.dport%,}"
    #     "{UDP:%UDP.sport%,%UDP.dport%}")
    if IP in x:
        ip_src = x[IP].src
        ip_dst = x[IP].dst
        ip_len = x[IP].len
        flow_time = x.time
        if TCP in x:
            conn = 0# used to identify if tcp or UDP
            src_port = x[TCP].sport
            dst_port = x[TCP].dport
        elif UDP in x:
            conn = 1
            src_port = x[UDP].sport
            dst_port = x[UDP].dport
        if TCP in x and ip_len > 50: # and size of packet is greater than 50
            storage.append([ip_src,src_port,ip_dst,dst_port,conn,flow_time,ip_len])
            if [ip_src,src_port,ip_dst,dst_port,conn] not in dataflow:#if not in dataflow append to it. Dataflow should only contain unique [ip_src,src_port,ip_dst,dst_port,conn]
                dataflow.append([ip_src,src_port,ip_dst,dst_port,conn])
        elif UDP in x and ip_len > 38:# if in UDP
            storage.append([ip_src,src_port,ip_dst,dst_port,conn,flow_time,ip_len])
            if [ip_src,src_port,ip_dst,dst_port,conn] not in dataflow:
                dataflow.append([ip_src,src_port,ip_dst,dst_port,conn])

        
def detectflow(storage):# trying to detect bidirectional network flow, not been tested
    size_list = []
    time_list = []
    print(*storage)
    print(len(dataflow))
    for v in range(len(dataflow)):# loop through dataflow list
        size_list = []
        time_list = []
        for i in range(len(storage)):# loop through storage list
            if dataflow[v][0] == storage[i][0] and dataflow[v][1] == storage[i][1] and dataflow[v][2] == storage[i][2] and dataflow[v][3] == storage[i][3]:# get packet size and time for packets sent from src to destination
                size_list.extend([storage[i][6]])
                time_list.extend([storage[i][5]])
            if dataflow[v][0] == storage[i][2] and dataflow[v][1] == storage[i][3] and dataflow[v][2] == storage[i][0] and dataflow[v][3] == storage[i][1]:# get packet size and time send from dst to source
                size_list.extend([storage[i][6]])
                time_list.extend([storage[i][5]])
        if sum(size_list) > 1000 : # get samples that have over 1000 in total packet size
            print(*size_list," -->",*time_list,"bi","\n") # testing purposes
            print(len(time_list))
            print(min(size_list),max(size_list),np.mean(size_list),np.std(size_list),"\n")
            print(max(time_list)-min(time_list),np.std(time_list),"\n")
            if [dataflow[v][4],min(size_list),max(size_list),np.mean(size_list),np.std(size_list),max(time_list)-min(time_list),np.std(time_list),sys.argv[1]] not in total_list:# make sure you don't have duplicates
                total_list.append([dataflow[v][4],min(size_list),max(size_list),np.mean(size_list),np.std(size_list),max(time_list)-min(time_list),np.std(time_list),sys.argv[1]])
        #print(*size_list," -->",*time_list)
        #rint(np.std(one_len),' std ',np.mean(one_len),' mean','one flow')

def writetofile(total_list):# write to csv file
    with open('test15.csv','a',newline="") as csvFile:
        writer = csv.writer(csvFile)
        for row in total_list:
            writer.writerow(row)
    csvFile.close()      

pkts = sniff(prn = fields_extraction, count = 100)
detectflow(storage)
writetofile(total_list)