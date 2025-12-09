# analyzer.py

import hashlib
import json
import time
from datetime import datetime
from typing import Dict, Any, List

# --- BLOCKCHAIN IMPLEMENTATION ---

class Block:
    """Represents a single block in the blockchain."""
    def __init__(self, index: int, previous_hash: str, data: Dict[str, Any], proof: int = 1):
        self.index = index
        self.timestamp = int(time.time())
        self.data = data
        self.previous_hash = previous_hash
        self.proof = proof  # Placeholder for Proof of Work/other validation
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """
        Computes the SHA-256 hash for the block content.
        The data must be serialized to ensure a consistent hash.
        """
        # We need a stable, ordered string representation of the block's data
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "proof": self.proof,
        }, sort_keys=True).encode()
        
        return hashlib.sha256(block_string).hexdigest()

class Blockchain:
    """Manages the chain of blocks."""
    def __init__(self):
        self.chain: List[Block] = []
        # Create the Genesis Block
        if not self.chain:
            self.create_genesis_block()

    def create_genesis_block(self):
        """Creates the first block in the chain."""
        self.chain.append(
            Block(
                index=0,
                previous_hash="0",  # Genesis block has no previous hash
                data={
                    "message": "Genesis Block: Binary Analysis Chain Initialized",
                    "file": "N/A",
                    "algo_id": 0,
                    "algo_name": "N/A",
                    "arch_name": "N/A",
                    "protocol_name": "N/A",
                    "is_proprietary": "N/A",
                    "overall_metrics": None,
                },
            )
        )

    def get_last_block(self) -> Block:
        """Returns the last block in the chain."""
        return self.chain[-1]

    def add_block(self, data: Dict[str, Any]) -> Block:
        """Creates a new block and adds it to the chain."""
        last_block = self.get_last_block()
        new_block = Block(
            index=last_block.index + 1,
            previous_hash=last_block.hash,
            data=data,
        )
        self.chain.append(new_block)
        return new_block
    
    # You would typically add a `is_chain_valid` method here

# --- MACHINE LEARNING MOCK FUNCTIONS ---

def predict_for_binary(filepath: str) -> Dict[str, Any]:
    """
    Simulates a machine learning prediction pipeline for a binary file.
    In a real system, this would extract features, load the model, and predict.
    """
    print(f"Analyzing file: {filepath}")
    
    # Mock Feature Extraction (e.g., extracting file size, entropy, import symbols)
    # In a real scenario, you'd use libraries like pefile, LIEF, or capstone
    input_features = {
        "file_size_bytes": 1024 + len(filepath),
        "entropy_score": 7.5,
        "section_count": 5,
    }

    # Mock Model Output
    algo_id = 15
    algo_name = "AES" if len(filepath) % 2 == 0 else "RSA"
    
    return {
        "file": filepath.split("/")[-1], # Simple filename
        "algo_id": algo_id,
        "algo_name": algo_name,
        "arch_name": "x86_64",
        "protocol_name": "TLSv1.3",
        "ocsvm_score": 0.98, # One-Class SVM score for anomaly/proprietary detection
        "ocsvm_is_proprietary": True if algo_name == "AES" else False,
        "ensemble_prediction": {"AES": 0.55, "RSA": 0.45},
        "input_features": input_features,
    }

def load_model_metrics() -> Dict[str, Any]:
    """
    Simulates loading stored model performance metrics.
    """
    # In a real MLOps pipeline, these metrics would be fetched from an artifact store (e.g., MLflow, Neptune)
    return {
        "overall_ensemble": {
            "accuracy": 0.95,
            "f1_score": 0.92,
            "precision": 0.94,
            "recall": 0.91,
        },
        "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# End of analyzer.py