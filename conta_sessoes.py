#BRUNO EDUARDO FARIAS
#BCC - CI1030 - 2021-2
#GRR20186715
import os
import sys
import logging
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
from scapy.all import *

def process_pcap(file_name):
    count = 0
    countIP = 0
    count_N_IP = 0
    countTCP = 0
    countUDP = 0

    count_sessoes_TCP = 0
    count_sessoes_UDP = 0

    #realiza a leitura do pcap utilizando a função rdpcap do scapy
    leitura_pcap = rdpcap(file_name)

    #percorre cada pacote e faz as contagens
    for packet in leitura_pcap:
        #contagem total de pacotes
        count += 1

        #contagem de pacotes IP e não-IP
        if packet.haslayer(IP) or packet.haslayer(IPv6):
            countIP += 1
        else:
            count_N_IP += 1

        #contagem de pacotes TCP
        if packet.haslayer(TCP):
            countTCP += 1

        #contagem de pacotes UDP
        if packet.haslayer(UDP):
            countUDP += 1

    #captura as sessões em dicionário
    sessions = leitura_pcap.sessions()

    for session in sessions:
        #contagem sessão UDP e TCP
        if session[0:3] == "UDP":
            count_sessoes_UDP += 1
        elif session[0:3] == "TCP":
            count_sessoes_TCP += 1

    #impressão dos resultados
    print(str(count) + ' pacotes no total')
    print(str(countIP) + ' pacotes IP')
    print(str(countTCP) + ' pacotes TCP')
    print(str(countUDP) + ' pacotes UDP')
    print(str(count_sessoes_TCP) + ' sessões TCP')
    print(str(count_sessoes_UDP) + ' sessões UDP')
    print(str(count_N_IP) + ' pacotes não-IP')

if __name__ == '__main__':
    nomeArquivo = "trace.pcap"
    if not os.path.isfile(nomeArquivo):
        print('"{}" não existe!'.format(nomeArquivo), file=sys.stderr)
        sys.exit(-1)

    process_pcap(nomeArquivo)
    sys.exit(0)