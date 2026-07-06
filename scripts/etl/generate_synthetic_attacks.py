import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)
N = 5000

def clip_int(arr, lo, hi):
    return np.clip(arr.round().astype(int), lo, hi)

def bernoulli(p, size):
    return (RNG.random(size) < p).astype(int)

def tcp_flags_value(syn, ack, rst, fin, psh):
    return (fin*0x01 + syn*0x02 + rst*0x04 + psh*0x08 + ack*0x10).astype(int)

def common_ip(size, proto=6):
    return (np.ones(size,dtype=int)*4, np.ones(size,dtype=int)*20,
            np.clip((RNG.normal(64,10,size)).round().astype(int),32,128),
            np.ones(size,dtype=int)*proto, np.zeros(size,dtype=int), np.ones(size,dtype=int))

def gen_brute_force(n):
    iv,ihl,ttl,ipr,l2,hip = common_ip(n,6)
    dst=RNG.choice([22,21],size=n,p=[0.75,0.25])
    src=RNG.integers(1025,65535,size=n)
    phase=RNG.choice([0,1],size=n,p=[0.6,0.4])
    syn=np.ones(n,dtype=int); ack=phase
    rst=bernoulli(0.03,n); fin=np.zeros(n,dtype=int); psh=np.zeros(n,dtype=int)
    pl=clip_int(RNG.normal(8,6,n),0,40)
    pe=RNG.uniform(0.5,2.5,n)
    pkl=clip_int(ihl+20+pl+clip_int(RNG.normal(4,2,n),0,10),40,150)
    return pd.DataFrame({"attack_type":"BRUTE_FORCE","label":1,"packet_length":pkl,"has_ip":hip,"has_tcp":np.ones(n,dtype=int),"has_udp":np.zeros(n,dtype=int),"has_icmp":np.zeros(n,dtype=int),"payload_length":pl,"payload_entropy":np.round(pe,4),"is_syn":syn,"is_ack":ack,"is_rst":rst,"is_fin":fin,"is_psh":psh,"is_high_port_src":np.ones(n,dtype=int),"is_high_port_dst":np.zeros(n,dtype=int),"is_well_known_port":np.ones(n,dtype=int),"ip_header_length":ihl,"tcp_window_size":np.ones(n,dtype=int)*8192,"packet_direction":np.ones(n,dtype=int),"ip_version":iv,"ip_ttl":ttl,"ip_proto":ipr,"src_port":src,"dst_port":dst,"tcp_flags":tcp_flags_value(syn,ack,rst,fin,psh),"is_layer2_only":l2})

def gen_slow_http(n):
    iv,ihl,ttl,ipr,l2,hip = common_ip(n,6)
    dst=RNG.choice([80,443],size=n,p=[0.5,0.5])
    src=RNG.integers(1025,65535,size=n)
    syn=np.zeros(n,dtype=int); ack=np.ones(n,dtype=int)
    rst=np.zeros(n,dtype=int); fin=bernoulli(0.01,n); psh=np.zeros(n,dtype=int)
    pl=clip_int(RNG.normal(20,10,n),1,50)
    pe=np.round(RNG.uniform(1.8,3.2,n),4)
    pkl=clip_int(ihl+20+pl,40,120)
    win=np.clip(np.full(n,65535,dtype=int)+RNG.integers(-128,128,size=n),32768,65535)
    return pd.DataFrame({"attack_type":"SLOW_HTTP","label":1,"packet_length":pkl,"has_ip":hip,"has_tcp":np.ones(n,dtype=int),"has_udp":np.zeros(n,dtype=int),"has_icmp":np.zeros(n,dtype=int),"payload_length":pl,"payload_entropy":pe,"is_syn":syn,"is_ack":ack,"is_rst":rst,"is_fin":fin,"is_psh":psh,"is_high_port_src":np.ones(n,dtype=int),"is_high_port_dst":np.zeros(n,dtype=int),"is_well_known_port":np.ones(n,dtype=int),"ip_header_length":ihl,"tcp_window_size":win,"packet_direction":np.ones(n,dtype=int),"ip_version":iv,"ip_ttl":ttl,"ip_proto":ipr,"src_port":src,"dst_port":dst,"tcp_flags":tcp_flags_value(syn,ack,rst,fin,psh),"is_layer2_only":l2})

def gen_dns_tunneling(n):
    iv,ihl,ttl,ipr,l2,hip = common_ip(n,17)
    dst=np.full(n,53,dtype=int)
    src=RNG.integers(1025,65535,size=n)
    syn=ack=rst=fin=psh=np.zeros(n,dtype=int)
    pl=clip_int(RNG.normal(220,75,n),100,400)
    pe=np.round(RNG.uniform(6.0,7.9,n),4)
    pkl=clip_int(ihl+8+pl,60,500)
    return pd.DataFrame({"attack_type":"DNS_TUNNELING","label":1,"packet_length":pkl,"has_ip":hip,"has_tcp":np.zeros(n,dtype=int),"has_udp":np.ones(n,dtype=int),"has_icmp":np.zeros(n,dtype=int),"payload_length":pl,"payload_entropy":pe,"is_syn":syn,"is_ack":ack,"is_rst":rst,"is_fin":fin,"is_psh":psh,"is_high_port_src":np.ones(n,dtype=int),"is_high_port_dst":np.zeros(n,dtype=int),"is_well_known_port":np.ones(n,dtype=int),"ip_header_length":ihl,"tcp_window_size":np.zeros(n,dtype=int),"packet_direction":RNG.choice([0,1],size=n,p=[0.45,0.55]),"ip_version":iv,"ip_ttl":ttl,"ip_proto":ipr,"src_port":src,"dst_port":dst,"tcp_flags":np.zeros(n,dtype=int),"is_layer2_only":l2})

def gen_port_scan(n):
    iv,ihl,ttl,ipr,l2,hip = common_ip(n,6)
    src=RNG.integers(40000,65535,size=n)
    dst=RNG.integers(1,65535,size=n)
    syn=np.ones(n,dtype=int); ack=rst=fin=psh=np.zeros(n,dtype=int)
    pl=np.zeros(n,dtype=int); pe=np.zeros(n)
    pkl=clip_int(ihl+20+clip_int(RNG.normal(2,2,n),0,8),40,64)
    win=clip_int(RNG.normal(1024,256,n),512,4096)
    wk=((dst==80)|(dst==443)|(dst==22)|(dst==21)|(dst==53)|(dst==25)).astype(int)
    return pd.DataFrame({"attack_type":"PORT_SCAN","label":1,"packet_length":pkl,"has_ip":hip,"has_tcp":np.ones(n,dtype=int),"has_udp":np.zeros(n,dtype=int),"has_icmp":np.zeros(n,dtype=int),"payload_length":pl,"payload_entropy":pe,"is_syn":syn,"is_ack":ack,"is_rst":rst,"is_fin":fin,"is_psh":psh,"is_high_port_src":np.ones(n,dtype=int),"is_high_port_dst":(dst>1024).astype(int),"is_well_known_port":wk,"ip_header_length":ihl,"tcp_window_size":win,"packet_direction":np.ones(n,dtype=int),"ip_version":iv,"ip_ttl":ttl,"ip_proto":ipr,"src_port":src,"dst_port":dst,"tcp_flags":tcp_flags_value(syn,ack,rst,fin,psh),"is_layer2_only":l2})

combined = pd.concat([gen_brute_force(N),gen_slow_http(N),gen_dns_tunneling(N),gen_port_scan(N)],ignore_index=True)
combined = combined.sample(frac=1,random_state=42).reset_index(drop=True)
combined.to_csv("enhanced_synthetic_attacks.csv",index=False)
print(f"✅ Saved {len(combined):,} rows → enhanced_synthetic_attacks.csv")
print(combined["attack_type"].value_counts().to_string())
