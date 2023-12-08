"""Implementation of the Flamingo protocol by Riccardo Taiello."""

from math import ceil
from typing import Dict, List, Optional

import numpy as np
from Crypto.Cipher import ChaCha20


class PRF:
    """
    Pseudo-Random Function (PRF) class for the Flamingo protocol.

    Args:
        vectorsize (int): The size of the vector to be generated.
        elementsize (int): The size of each element in the vector in bits.

    Attributes:
        vectorsize (int): The size of the vector to be generated.
        bits_ptxt (int): The size of each element in the vector in bits.
        num_bytes (int): The size of each element in the vector in bytes.

    Methods:
        eval_key(key: bytes, round: int) -> bytes:
            Evaluates the PRF on the given key and round.

        eval_vector(seed) -> bytes:
            Evaluates the PRF on the given seed to generate a random vector.
    """

    _zero = 0
    _nonce = _zero.to_bytes(12, "big")
    security = 256

    def __init__(self, vectorsize: int, elementsize: int) -> None:
        super().__init__()
        self.vectorsize = vectorsize
        self.bits_ptxt = elementsize
        self.num_bytes = ceil(elementsize / 8)

    def eval_key(self, key: bytes, round: int) -> bytes:
        """
        Evaluates the PRF on the given key and round.

        Args:
            key (bytes): The key for the PRF.
            round (int): The round number.

        Returns:
            bytes: The output of the PRF.
        """
        round_number_bytes = round.to_bytes(16, "big")
        c = ChaCha20.new(key=key, nonce=PRF._nonce).encrypt(round_number_bytes)
        c = c + b"\x00" * 16  # Pad the output to 32 bytes
        return c

    def eval_vector(self, seed: bytes) -> bytes:
        """
        Evaluates the PRF on the given seed to generate a random vector.

        Args:
            seed (bytes): The seed for the PRF.

        Returns:
            bytes: The generated random vector.
        """
        c = ChaCha20.new(key=seed, nonce=PRF._nonce)
        data = b"secr" * self.vectorsize
        return c.encrypt(data)


class Flamingo:
    """
    Flamingo protocol implementation.

    Attributes:
        prf (Optional[PRF]): An instance of the PRF class.
        vector_dtype (str): The data type of the vectors.

    Methods:
        setup_pairwise_secrets(my_node_id: int, nodes_ids: List[str], num_params: int) -> None:
            Sets up pairwise secrets for communication.

        protect(current_round: int, params: List[int], node_ids: List[str]) -> List[int]:
            Protects the given parameters using the Flamingo protocol.
    """

    prf: Optional[PRF] = None

    def __init__(self) -> None:
        super().__init__()
        self.vector_dtype = "uint32"

    def setup_pairwise_secrets(
        self, my_node_id: int, nodes_ids: List[str], num_params: int
    ) -> None:
        """
        Sets up pairwise secrets for communication.

        Args:
            my_node_id (int): The identifier of the current node.
            nodes_ids (List[str]): The list of identifiers of all nodes.
            num_params (int): The number of parameters.

        Returns:
            None
        """
        Flamingo.prf = PRF(vectorsize=num_params, elementsize=16)
        self.pairwise_secrets = {}
        self.pairwise_seeds = {}
        self.my_node_id = my_node_id

        for node_id in nodes_ids:
            if node_id != my_node_id:
                with open(
                    f"dh_certificates/shared_secret_{node_id}_{my_node_id}.bin", "rb"
                ) as f:
                    self.pairwise_secrets[node_id] = f.read()[:32]

    def protect(
        self,
        current_round: int,
        params: List[int],
        node_ids: List[str],
    ) -> List[int]:
        """
        Protects the given parameters using the Flamingo protocol.

        Args:
            current_round (int): The current round number.
            params (List[int]): The parameters to be protected.
            node_ids (List[str]): The list of node identifiers.

        Returns:
            List[int]: The protected parameters.
        """
        params = np.array(params, dtype=self.vector_dtype)
        vec = np.zeros(len(params), dtype=self.vector_dtype)

        for node_id in node_ids:
            if node_id == self.my_node_id:
                continue
            secret = self.pairwise_secrets[node_id]
            pairwise_seed = Flamingo.prf.eval_key(key=secret, round=current_round)
            pairwise_vector = Flamingo.prf.eval_vector(seed=pairwise_seed)
            pairwise_vector = np.frombuffer(pairwise_vector, dtype=self.vector_dtype)
            if node_id < self.my_node_id:
                vec += pairwise_vector
            else:
                vec -= pairwise_vector

        encrypted_params = vec + params
        return encrypted_params.tolist()
