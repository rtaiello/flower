#!/bin/bash

# Number of clients
N=2

# Generate DH parameters
openssl dhparam -out dhparam.pem 2048

# Loop through clients
for ((i=0; i<N; i++)); do
    # Generate private key
    openssl genpkey -paramfile dhparam.pem -out private_key_$i.pem

    # Extract public key from the private key
    openssl pkey -in private_key_$i.pem -pubout -out public_key_$i.pem
done

# Pairwise key exchange
for ((i=0; i<N; i++)); do
    for ((j=i+1; j<N; j++)); do
        # Perform key exchange between client i and client j
        openssl pkeyutl -derive -inkey private_key_$i.pem -peerkey public_key_$j.pem -out shared_secret_${i}_${j}.bin
        openssl pkeyutl -derive -inkey private_key_$j.pem -peerkey public_key_$i.pem -out shared_secret_${j}_${i}.bin
    done
done

# Display the shared secrets
for ((i=0; i<N; i++)); do
    for ((j=i+1; j<N; j++)); do
        echo "Shared secret between client $i and client $j:"
        head -c 32 shared_secret_${i}_${j}.bin | xxd -p
        echo
    done
done
