"""Chemical RAG system — lightweight vector search + knowledge graph.

Uses sklearn TF-IDF for embeddings (no API keys, no heavy downloads)
and NetworkX for chemical knowledge graph with typed edges.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import AutoChemConfig
from src.data.models import ReactionHypothesis, VerificationResult, VerificationStatus


class ChemicalKnowledgeGraph:
    """Directed chemical knowledge graph with typed edges."""

    EDGE_TYPES = [
        "reactant_of",
        "product_of",
        "has_scaffold",
        "contains_group",
        "similar_to",
    ]

    def __init__(self):
        self._graph = nx.MultiDiGraph()
        self._node_count = 0
        self._edge_count = 0

    def add_reaction(
        self,
        reaction_id: str,
        reactants: list[str],
        products: list[str],
        reaction_type: str,
    ) -> None:
        reaction_node = f"rxn:{reaction_id}"
        self._graph.add_node(reaction_node, type="reaction", reaction_type=reaction_type)
        self._node_count += 1

        for smi in reactants:
            node = f"mol:{smi}"
            if node not in self._graph:
                self._graph.add_node(node, type="molecule", smiles=smi)
                self._node_count += 1
            self._graph.add_edge(node, reaction_node, type="reactant_of")
            self._edge_count += 1

        for smi in products:
            node = f"mol:{smi}"
            if node not in self._graph:
                self._graph.add_node(node, type="molecule", smiles=smi)
                self._node_count += 1
            self._graph.add_edge(reaction_node, node, type="product_of")
            self._edge_count += 1

    def add_scaffold(self, smiles: str, scaffold_smiles: str) -> None:
        mol_node = f"mol:{smiles}"
        scaff_node = f"scaffold:{scaffold_smiles}"
        if mol_node not in self._graph:
            self._graph.add_node(mol_node, type="molecule", smiles=smiles)
            self._node_count += 1
        if scaff_node not in self._graph:
            self._graph.add_node(scaff_node, type="scaffold", smiles=scaffold_smiles)
            self._node_count += 1
        self._graph.add_edge(mol_node, scaff_node, type="has_scaffold")
        self._edge_count += 1

    def add_functional_group(self, smiles: str, group_name: str) -> None:
        mol_node = f"mol:{smiles}"
        group_node = f"group:{group_name}"
        if mol_node not in self._graph:
            self._graph.add_node(mol_node, type="molecule", smiles=smiles)
            self._node_count += 1
        if group_node not in self._graph:
            self._graph.add_node(group_node, type="functional_group", name=group_name)
            self._node_count += 1
        self._graph.add_edge(mol_node, group_node, type="contains_group")
        self._edge_count += 1

    def get_neighbors(self, node_id: str, depth: int = 1) -> list[str]:
        if node_id not in self._graph:
            return []
        neighbors = set()
        current = {node_id}
        for _ in range(depth):
            next_level = set()
            for n in current:
                for neighbor in self._graph.neighbors(n):
                    if neighbor not in neighbors:
                        next_level.add(neighbor)
            neighbors.update(next_level)
            current = next_level
        return list(neighbors)

    def query_reactions_by_molecule(self, smiles: str) -> list[str]:
        node = f"mol:{smiles}"
        if node not in self._graph:
            return []
        reactions = []
        for _, target, data in self._graph.out_edges(node, data=True):
            if data.get("type") == "reactant_of":
                reactions.append(target)
        for source, _, data in self._graph.in_edges(node, data=True):
            if data.get("type") == "product_of":
                reactions.append(source)
        return reactions

    def status(self) -> dict:
        return {
            "nodes": self._node_count,
            "edges": self._edge_count,
            "initialized": self._graph is not None,
        }


class ChemicalRAG:
    """Retrieval-augmented generation using TF-IDF + chemical knowledge graph.

    No API keys required — uses sklearn TF-IDF for embeddings and
    NetworkX for the knowledge graph. Fast, lightweight, offline.
    """

    def __init__(self, config: AutoChemConfig):
        self.config = config
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._documents: list[str] = []
        self._document_vectors: Optional[np.ndarray] = None
        self._initialized = False
        self._knowledge_graph = ChemicalKnowledgeGraph()
        self._persist_dir = Path(config.rag.chroma_persist_dir)

    def initialize(self) -> bool:
        if self._initialized:
            return True
        try:
            self._vectorizer = TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                analyzer="word",
                ngram_range=(1, 2),
                max_features=5000,
            )
            self._load_state()
            self._initialized = True
            logger.info("Chemical RAG initialized (TF-IDF mode): {} docs",
                        len(self._documents))
            return True
        except Exception as e:
            logger.error("Failed to initialize RAG: {}", e)
            return False

    @property
    def knowledge_graph(self) -> ChemicalKnowledgeGraph:
        return self._knowledge_graph

    def index_reaction(
        self,
        hypothesis: ReactionHypothesis,
        result: VerificationResult,
    ) -> None:
        if not self._initialized:
            return
        reactants = ", ".join(r.smiles for r in hypothesis.reactants)
        products = ", ".join(p.smiles for p in hypothesis.products)
        doc = (
            f"Reaction type: {hypothesis.reaction_type.value}. "
            f"Reactants: {reactants}. "
            f"Products: {products}. "
            f"Status: {result.status.value}. "
            f"Yield: {hypothesis.yield_estimate or 'unknown'}%. "
        )
        if hypothesis.mechanism_steps:
            doc += f"Mechanism: {hypothesis.mechanism_steps[:200]}. "
        self._documents.append(doc)
        self._recompute_vectors()

        self._knowledge_graph.add_reaction(
            reaction_id=str(hypothesis.id),
            reactants=[r.smiles for r in hypothesis.reactants],
            products=[p.smiles for p in hypothesis.products],
            reaction_type=hypothesis.reaction_type.value,
        )

    def _recompute_vectors(self) -> None:
        if not self._documents:
            self._document_vectors = None
            return
        try:
            self._document_vectors = self._vectorizer.fit_transform(self._documents)
        except Exception:
            self._document_vectors = None

    def retrieve_context(
        self,
        query: str,
        k: int | None = None,
    ) -> list[str]:
        if not self._initialized or not self._documents:
            return []
        k = k or self.config.rag.retrieval_k
        try:
            query_vec = self._vectorizer.transform([query])
            if self._document_vectors is None:
                return []
            from sklearn.metrics.pairwise import cosine_similarity
            sims = cosine_similarity(query_vec, self._document_vectors).flatten()
            top_indices = np.argsort(sims)[-k:][::-1]
            results = []
            for idx in top_indices:
                if sims[idx] > 0.01:
                    results.append(self._documents[idx])
            return results[:k]
        except Exception as e:
            logger.warning("RAG retrieval failed: {}", e)
            return []

    def enrich_prompt(
        self,
        base_prompt: str,
        reaction_type: str | None = None,
    ) -> str:
        if not self._initialized:
            return base_prompt
        context_docs = self.retrieve_context(base_prompt, k=self.config.rag.retrieval_k)
        if not context_docs:
            return base_prompt
        context = "\n".join(f"- {doc[:200]}" for doc in context_docs)
        return (
            f"Previous reaction data from the knowledge base:\n{context}\n\n"
            f"Using the above as reference, {base_prompt}"
        )

    def multi_hop_retrieve(
        self,
        query: str,
        hop_depth: int = 2,
    ) -> list[str]:
        if not self._initialized:
            return []
        direct_results = self.retrieve_context(query, k=3)
        from src.chemistry.rdkit_wrapper import smiles_to_mol
        all_results = list(direct_results)
        smiles_pattern = __import__("re").compile(r'[A-Za-z0-9@+\-\[\]\(\)\\/%=#\.]+')
        for doc in direct_results:
            potential_smiles = smiles_pattern.findall(doc)
            for smi in potential_smiles[:3]:
                if len(smi) < 3:
                    continue
                neighbors = self._knowledge_graph.get_neighbors(f"mol:{smi}", depth=hop_depth)
                for neighbor in neighbors[:3]:
                    all_results.append(f"Related molecule/reaction: {neighbor}")
        return all_results[:10]

    def _load_state(self) -> None:
        state_file = self._persist_dir / "rag_state.pkl"
        if state_file.exists():
            try:
                with open(state_file, "rb") as f:
                    state = pickle.load(f)
                self._documents = state.get("documents", [])
                if self._documents:
                    self._recompute_vectors()
            except Exception:
                pass

    def _save_state(self) -> None:
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        state_file = self._persist_dir / "rag_state.pkl"
        try:
            with open(state_file, "wb") as f:
                pickle.dump({"documents": self._documents}, f)
        except Exception:
            pass

    def status(self) -> dict:
        kg_status = self._knowledge_graph.status()
        return {
            "initialized": self._initialized,
            "documents_indexed": len(self._documents),
            "vector_dim": self._document_vectors.shape[1] if self._document_vectors is not None else 0,
            "knowledge_graph_nodes": kg_status["nodes"],
            "knowledge_graph_edges": kg_status["edges"],
            "persist_dir": str(self._persist_dir),
        }


def extract_smiles_from_text(text: str) -> list[str]:
    import re
    pattern = re.compile(
        r'(?:(?:[A-Z][a-z]?|\[[^\]]+\])[0-9]*(?:[=#/\\@+\-]?){0,2})+'
    )
    return pattern.findall(text)
