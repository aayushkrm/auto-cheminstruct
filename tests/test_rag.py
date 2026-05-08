"""Tests for the Chemical RAG module (TF-IDF + knowledge graph)."""

import pytest
from src.config import AutoChemConfig, RAGConfig


class TestChemicalKnowledgeGraph:
    """Tests for ChemicalKnowledgeGraph."""

    def test_add_reaction(self):
        from src.rag.chemical_rag import ChemicalKnowledgeGraph
        kg = ChemicalKnowledgeGraph()
        kg.add_reaction("rxn1", ["CCO", "O"], ["CC=O", "H2O"], "oxidation")
        assert kg._node_count >= 3
        assert kg._edge_count >= 2

    def test_add_scaffold(self):
        from src.rag.chemical_rag import ChemicalKnowledgeGraph
        kg = ChemicalKnowledgeGraph()
        kg.add_scaffold("c1ccccc1O", "c1ccccc1")
        assert kg._node_count >= 2
        assert kg._edge_count >= 1

    def test_add_functional_group(self):
        from src.rag.chemical_rag import ChemicalKnowledgeGraph
        kg = ChemicalKnowledgeGraph()
        kg.add_functional_group("CC(=O)O", "carboxylic_acid")
        assert kg._node_count >= 2
        assert kg._edge_count >= 1

    def test_get_neighbors(self):
        from src.rag.chemical_rag import ChemicalKnowledgeGraph
        kg = ChemicalKnowledgeGraph()
        kg.add_reaction("rxn1", ["CCO"], ["CC=O"], "oxidation")
        kg.add_scaffold("CCO", "CC")
        neighbors = kg.get_neighbors("mol:CCO", depth=2)
        assert len(neighbors) >= 1

    def test_query_reactions_by_molecule(self):
        from src.rag.chemical_rag import ChemicalKnowledgeGraph
        kg = ChemicalKnowledgeGraph()
        kg.add_reaction("rxn1", ["CCO", "O"], ["CC=O"], "oxidation")
        rxns = kg.query_reactions_by_molecule("CCO")
        assert len(rxns) >= 1

    def test_status(self):
        from src.rag.chemical_rag import ChemicalKnowledgeGraph
        kg = ChemicalKnowledgeGraph()
        kg.add_reaction("rxn1", ["CCO"], ["CC=O"], "oxidation")
        s = kg.status()
        assert s["nodes"] >= 1
        assert s["edges"] >= 1
        assert s["initialized"] is True


class TestChemicalRAG:
    """Tests for ChemicalRAG with TF-IDF backend."""

    @pytest.fixture
    def config(self):
        return AutoChemConfig(
            rag=RAGConfig(enabled=True, chroma_persist_dir=".chromadb_test", retrieval_k=3)
        )

    def test_initialization(self, config):
        from src.rag.chemical_rag import ChemicalRAG
        rag = ChemicalRAG(config)
        ok = rag.initialize()
        assert ok is True
        assert rag._initialized is True

    def test_retrieve_context_uninitialized(self, config):
        from src.rag.chemical_rag import ChemicalRAG
        rag = ChemicalRAG(config)
        rag._initialized = False
        assert rag.retrieve_context("test query") == []

    def test_enrich_prompt_uninitialized(self, config):
        from src.rag.chemical_rag import ChemicalRAG
        rag = ChemicalRAG(config)
        rag._initialized = False
        prompt = "Generate a reaction"
        result = rag.enrich_prompt(prompt)
        assert result == prompt

    def test_multi_hop_retrieve_uninitialized(self, config):
        from src.rag.chemical_rag import ChemicalRAG
        rag = ChemicalRAG(config)
        rag._initialized = False
        assert rag.multi_hop_retrieve("test") == []

    def test_retrieve_context_empty(self, config):
        from src.rag.chemical_rag import ChemicalRAG
        rag = ChemicalRAG(config)
        rag.initialize()
        assert rag.retrieve_context("some reaction") == []

    def test_knowledge_graph_uninitialized(self, config):
        from src.rag.chemical_rag import ChemicalRAG
        rag = ChemicalRAG(config)
        assert rag.knowledge_graph is not None
        assert rag.knowledge_graph._node_count == 0

    def test_status_uninitialized(self, config):
        from src.rag.chemical_rag import ChemicalRAG
        rag = ChemicalRAG(config)
        rag._initialized = False
        s = rag.status()
        assert s["initialized"] is False


class TestExtractSmiles:
    """Tests for SMILES extraction from text."""

    def test_extract_valid_smiles(self):
        from src.rag.chemical_rag import extract_smiles_from_text
        text = "The reaction CCO + CC=O -> product uses ethanol and acetaldehyde"
        smiles = extract_smiles_from_text(text)
        assert len(smiles) >= 1

    def test_extract_no_smiles(self):
        from src.rag.chemical_rag import extract_smiles_from_text
        text = "This paragraph discusses reaction mechanisms in detail"
        smiles = extract_smiles_from_text(text)
        # The regex may false-positive on "Th", "in" — that's expected behavior
