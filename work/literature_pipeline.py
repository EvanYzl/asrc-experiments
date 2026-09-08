#!/usr/bin/env python3
"""Build an open-access literature pack for the QURA-Cert project.

The script resolves bibliographic metadata from OpenAlex/Crossref, downloads only
publicly reachable PDF files, validates the PDF signature, and writes a BibTeX
library plus human-readable manifests.  Publisher pages that do not expose a
legal public PDF are retained in the manual-download list.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import re
import shutil
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "literature"
PAPERS = OUT / "papers"
USER_AGENT = "QURA-Cert-literature-pack/1.0 (academic literature retrieval)"
CONTACT = "codex.research@example.com"


def load_previous() -> dict[str, dict[str, Any]]:
    path = OUT / "literature_manifest.json"
    if not path.exists():
        return {}
    try:
        return {row["key"]: row for row in json.loads(path.read_text(encoding="utf-8"))}
    except Exception:
        return {}


PREVIOUS = load_previous()


@dataclass
class Seed:
    key: str
    theme: str
    title: str
    importance: str = "supporting"
    year: int | None = None
    doi: str | None = None
    arxiv: str | None = None
    landing_url: str | None = None
    pdf_urls: list[str] = field(default_factory=list)
    local_source: str | None = None
    relevance: str = ""


def S(key: str, theme: str, title: str, **kwargs: Any) -> Seed:
    return Seed(key=key, theme=theme, title=title, **kwargs)


SEEDS: list[Seed] = [
    # 01 — Direct multi-domain / multilingual KGC lineage.
    S("dmkgc", "01_core_multidomain_kgc", "Conditional Diffusion Guided Knowledge Transfer for Multi-Domain Knowledge Graph Completion", importance="core", year=2026, doi="10.1145/3774904.3792252", arxiv="2607.03154", local_source="work/DMKGC-review/paper/Conditional Diffusion Guided Knowledge Transfer for Multi-Domain Knowledge Graph Completion.pdf", relevance="Closest generative multi-domain KGC competitor and shared benchmark source."),
    S("imkgc", "01_core_multidomain_kgc", "Information-Theoretic Minimal Sufficient Representation for Multi-Domain Knowledge Graph Completion", importance="core", year=2026, doi="10.1609/aaai.v40i18.38601", local_source="work/IMKGC-review/Paper_AAAI2026.pdf", relevance="Closest information-compression multi-domain KGC competitor."),
    S("imkgc_appendix", "01_core_multidomain_kgc", "Information-Theoretic Minimal Sufficient Representation for Multi-Domain Knowledge Graph Completion — Appendix", importance="core", year=2026, local_source="work/IMKGC-review/Appendix.pdf", landing_url="https://github.com/JiaweiSheng/IMKGC", relevance="Official supplementary material for IMKGC reproduction."),
    S("lsmga", "01_core_multidomain_kgc", "Multilingual Knowledge Graph Completion with Language-Sensitive Multi-Graph Attention", importance="core", year=2023, doi="10.18653/v1/2023.acl-long.586", pdf_urls=["https://aclanthology.org/2023.acl-long.586.pdf"], relevance="Attention-based cross-KG fusion baseline."),
    S("ssaga", "01_core_multidomain_kgc", "Multilingual Knowledge Graph Completion with Self-Supervised Adaptive Graph Alignment", importance="core", year=2022, doi="10.18653/v1/2022.acl-long.36", arxiv="2203.14987", pdf_urls=["https://aclanthology.org/2022.acl-long.36.pdf"], relevance="Alignment-edge attention and E-PKG benchmark source."),
    S("dual_spirals", "01_core_multidomain_kgc", "Learning Low-dimensional Multi-domain Knowledge Graph Embedding via Dual Archimedean Spirals", importance="core", year=2024, doi="10.18653/v1/2024.findings-acl.118", pdf_urls=["https://aclanthology.org/2024.findings-acl.118.pdf"], relevance="Low-dimensional geometric separation baseline for multi-domain KGs."),
    S("joint_pretrain_local_retrain", "01_core_multidomain_kgc", "Joint Pre-training and Local Re-training: Transferable Representation Learning on Multi-source Knowledge Graphs", importance="core", year=2023, doi="10.1145/3580305.3599397", arxiv="2306.02679", relevance="Transferable representation learning across multiple source KGs."),
    S("kens", "01_core_multidomain_kgc", "Multilingual Knowledge Graph Completion via Ensemble Knowledge Transfer", importance="core", year=2020, doi="10.18653/v1/2020.findings-emnlp.290", arxiv="2010.03158", pdf_urls=["https://aclanthology.org/2020.findings-emnlp.290.pdf"], relevance="Foundational multilingual KGC benchmark and DBP-5L source."),
    S("alignkgc", "01_core_multidomain_kgc", "Joint Completion and Alignment of Multilingual Knowledge Graphs", importance="core", year=2022, doi="10.18653/v1/2022.emnlp-main.817", arxiv="2104.08804", pdf_urls=["https://aclanthology.org/2022.emnlp-main.817.pdf"], relevance="Joint KGC/entity/relation alignment baseline."),
    S("joint_mkgc_alignment", "01_core_multidomain_kgc", "Joint Multilingual Knowledge Graph Completion and Alignment", importance="core", year=2022, doi="10.18653/v1/2022.findings-emnlp.341", arxiv="2210.08922", pdf_urls=["https://aclanthology.org/2022.findings-emnlp.341.pdf"], relevance="Alternative joint completion-and-alignment formulation."),
    S("ckgc_ckd", "01_core_multidomain_kgc", "Collective Knowledge Graph Completion with Mutual Knowledge Distillation", importance="core", year=2023, arxiv="2305.15895", relevance="Fused/global and local KG mutual-distillation transfer."),
    S("atransn", "01_core_multidomain_kgc", "An Adversarial Transfer Network for Knowledge Representation Learning", importance="core", year=2021, doi="10.1145/3442381.3450064", arxiv="2104.14757", relevance="Adversarial source-to-target KG embedding transfer baseline."),
    S("kd_mkb", "01_core_multidomain_kgc", "Knowledge Base Embedding By Cooperative Knowledge Distillation", importance="core", year=2020, doi="10.18653/v1/2020.coling-main.489", pdf_urls=["https://aclanthology.org/2020.coling-main.489.pdf"], relevance="Cooperative distillation across knowledge bases."),
    S("cgmua", "01_core_multidomain_kgc", "Collective Multi-type Entity Alignment Between Knowledge Graphs", importance="core", year=2020, doi="10.1145/3366423.3380289", pdf_urls=["https://cdn.amazon.science/ff/7a/b96282984a0fbe5e31a8fcf68d17/scipub-1202.pdf", "http://hanj.cs.illinois.edu/pdf/www20_qzhu.pdf"], relevance="Collective alignment/fusion baseline used by multi-domain KGC studies."),
    S("mtranse", "01_core_multidomain_kgc", "Multilingual Knowledge Graph Embeddings for Cross-lingual Knowledge Alignment", importance="core", year=2017, doi="10.24963/ijcai.2017/209", pdf_urls=["https://www.ijcai.org/proceedings/2017/0209.pdf"], relevance="Foundational aligned multilingual KG embedding method."),
    S("multiview_ea", "01_core_multidomain_kgc", "Multi-view Knowledge Graph Embedding for Entity Alignment", importance="core", year=2019, doi="10.24963/ijcai.2019/754", pdf_urls=["https://www.ijcai.org/proceedings/2019/0754.pdf"], relevance="Multi-view entity alignment used in cross-KG settings."),
    S("bootea", "01_core_multidomain_kgc", "Bootstrapping Entity Alignment with Knowledge Graph Embedding", importance="core", year=2018, doi="10.24963/ijcai.2018/611", pdf_urls=["https://www.ijcai.org/proceedings/2018/0611.pdf"], relevance="DWY benchmark lineage and bootstrapped alignment."),
    S("rnm", "01_core_multidomain_kgc", "Relation-Aware Neighborhood Matching Model for Entity Alignment", year=2021, doi="10.1609/aaai.v35i5.16606", arxiv="2012.08128", relevance="Relation-aware neighborhood evidence for aligned entities."),
    S("similarity_flooding", "01_core_multidomain_kgc", "What Makes Entities Similar? A Similarity Flooding Perspective for Multi-sourced Knowledge Graph Embeddings", year=2023, pdf_urls=["https://proceedings.mlr.press/v202/sun23d/sun23d.pdf"], landing_url="https://proceedings.mlr.press/v202/sun23d.html", relevance="Multi-source KG embedding and cross-graph similarity propagation."),
    S("cross_network_gcn", "01_core_multidomain_kgc", "Cross-Network Learning with Partially Aligned Graph Convolutional Networks", year=2021, arxiv="2106.01583", relevance="Theory and mechanisms for positive transfer across partially aligned graphs."),
    S("incomplete_kga", "01_core_multidomain_kgc", "Incomplete Knowledge Graph Alignment", year=2021, arxiv="2112.09266", relevance="Robust alignment when both graphs and correspondences are incomplete."),
    S("global_local_mkgc", "01_core_multidomain_kgc", "Multilingual Knowledge Graph Completion based on Global-Local Structure Encoding", year=2024, doi="10.1109/ISCTIS63324.2024.10699111", relevance="Recent global/local structure multi-KG baseline."),
    S("simplified_multiview_mkgc", "01_core_multidomain_kgc", "Simplified Multi-view Graph Neural Network for Multilingual Knowledge Graph Completion", year=2024, doi="10.1007/s11704-024-3577-3", pdf_urls=["https://journal.hep.com.cn/fcs/EN/PDF/10.1007/s11704-024-3577-3"], relevance="Recent lightweight multi-view multilingual KGC method."),
    S("efficient_multilingual_sharing", "01_core_multidomain_kgc", "Multilingual Knowledge Graph Completion via Efficient Multilingual Knowledge Sharing", importance="core", year=2025, doi="10.18653/v1/2025.findings-emnlp.577", arxiv="2510.07736", pdf_urls=["https://aclanthology.org/2025.findings-emnlp.577.pdf"], relevance="Recent source-sharing and mixture-of-experts multilingual KGC competitor."),
    S("plm_mkgc_constraints", "01_core_multidomain_kgc", "Multilingual Knowledge Graph Completion from Pretrained Language Models with Knowledge Constraints", year=2024, arxiv="2406.18085", relevance="Textual multilingual KGC with explicit KG constraints."),
    S("crosskg_substructure", "01_core_multidomain_kgc", "Cross-KG Link Prediction by Learning Substructural Semantics", year=2024, doi="10.1007/s11063-024-11537-9", relevance="Cross-KG link prediction from transferable substructures."),
    S("mkprompt", "01_core_multidomain_kgc", "Multi-domain Knowledge Graph Collaborative Pre-training and Prompt Tuning for Diverse Downstream Tasks", year=2024, arxiv="2405.13085", relevance="Collaborative multi-domain KG pretraining and adaptation."),
    S("daea", "01_core_multidomain_kgc", "DAEA: Enhancing Entity Alignment in Real-World Knowledge Graphs Through Multi-Source Domain Adaptation", year=2025, doi="10.18653/v1/2025.coling-main.393", pdf_urls=["https://aclanthology.org/2025.coling-main.393.pdf"], relevance="Explicit multi-source domain adaptation for entity alignment."),
    S("synergykgc", "01_core_multidomain_kgc", "SynergyKGC: Reconciling Topological Heterogeneity in Knowledge Graph Completion via Topology-Aware Synergy", importance="core", year=2026, arxiv="2602.10845", relevance="Query/structure-aware gated fusion adjacent to the proposed admission mechanism."),

    # 02 — KGC scorers, graph reasoners, and locally reproducible baselines.
    S("transe", "02_kgc_backbones_and_reasoning", "Translating Embeddings for Modeling Multi-relational Data", importance="core", year=2013, pdf_urls=["https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf"], landing_url="https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html", relevance="Lightweight decoder used by DMKGC and a strong local anchor."),
    S("distmult", "02_kgc_backbones_and_reasoning", "Embedding Entities and Relations for Learning and Inference in Knowledge Bases", importance="core", year=2015, arxiv="1412.6575", pdf_urls=["https://openreview.net/pdf?id=B1GmA3le"], relevance="Lightweight bilinear decoder and target-only candidate."),
    S("complex", "02_kgc_backbones_and_reasoning", "Complex Embeddings for Simple Link Prediction", importance="core", year=2016, arxiv="1606.06357", pdf_urls=["https://proceedings.mlr.press/v48/trouillon16.pdf"], relevance="Efficient asymmetric bilinear KGC baseline."),
    S("complex_jmlr", "02_kgc_backbones_and_reasoning", "Knowledge Graph Completion via Complex Tensor Factorization", year=2017, pdf_urls=["https://jmlr.csail.mit.edu/papers/volume18/16-563/16-563.pdf"], landing_url="https://jmlr.org/papers/v18/16-563.html", relevance="Extended ComplEx analysis and evaluation."),
    S("conve", "02_kgc_backbones_and_reasoning", "Convolutional 2D Knowledge Graph Embeddings", year=2018, doi="10.1609/aaai.v32i1.11573", arxiv="1707.01476", relevance="Influential neural KGC decoder and WN18RR/FB15k-237 reference."),
    S("rotate", "02_kgc_backbones_and_reasoning", "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space", importance="core", year=2019, arxiv="1902.10197", pdf_urls=["https://openreview.net/pdf?id=HkgEQnRqYQ"], relevance="Canonical relation-pattern KGC baseline."),
    S("tucker", "02_kgc_backbones_and_reasoning", "TuckER: Tensor Factorization for Knowledge Graph Completion", year=2019, doi="10.18653/v1/D19-1522", arxiv="1901.09590", pdf_urls=["https://aclanthology.org/D19-1522.pdf"], relevance="Strong tensor-factorization baseline."),
    S("rgcn", "02_kgc_backbones_and_reasoning", "Modeling Relational Data with Graph Convolutional Networks", importance="core", year=2018, doi="10.1007/978-3-319-93417-4_38", arxiv="1703.06103", relevance="Canonical relational message-passing encoder."),
    S("sacn", "02_kgc_backbones_and_reasoning", "End-to-End Structure-Aware Convolutional Networks for Knowledge Base Completion", year=2019, doi="10.1609/aaai.v33i01.33013060", arxiv="1811.04441", relevance="Structure-aware GCN KGC baseline."),
    S("compgcn", "02_kgc_backbones_and_reasoning", "Composition-based Multi-Relational Graph Convolutional Networks", importance="core", year=2020, arxiv="1911.03082", pdf_urls=["https://openreview.net/pdf?id=BylA_C4tPr"], relevance="Encoder used by collective multi-KG completion methods."),
    S("pairre", "02_kgc_backbones_and_reasoning", "PairRE: Knowledge Graph Embeddings via Paired Relation Vectors", year=2021, doi="10.18653/v1/2021.acl-long.336", arxiv="2011.03798", pdf_urls=["https://aclanthology.org/2021.acl-long.336.pdf"], relevance="Parameter-efficient embedding baseline."),
    S("hake", "02_kgc_backbones_and_reasoning", "Learning Hierarchy-Aware Knowledge Graph Embeddings for Link Prediction", year=2020, doi="10.1609/aaai.v34i03.5701", arxiv="1911.09419", relevance="Hierarchy-aware lightweight embedding baseline."),
    S("nbfnet", "02_kgc_backbones_and_reasoning", "Neural Bellman-Ford Networks: A General Graph Neural Network Framework for Link Prediction", importance="core", year=2021, arxiv="2106.06935", pdf_urls=["https://proceedings.neurips.cc/paper/2021/file/f6a673f09493afcd8b129a0bcf1cd5bc-Paper.pdf"], relevance="Query-conditioned path reasoning baseline."),
    S("redgnn", "02_kgc_backbones_and_reasoning", "Knowledge Graph Reasoning with Relational Digraph", year=2022, arxiv="2108.06040", relevance="Efficient relation-aware subgraph reasoning baseline."),
    S("astarnet", "02_kgc_backbones_and_reasoning", "A*Net: A Scalable Path-based Reasoning Approach for Knowledge Graphs", year=2023, arxiv="2206.04798", pdf_urls=["https://proceedings.neurips.cc/paper_files/paper/2023/file/b9e98316cb72fee82cc1160da5810abc-Paper-Conference.pdf"], relevance="Scalable query-dependent path selection."),
    S("ultra", "02_kgc_backbones_and_reasoning", "Towards Foundation Models for Knowledge Graph Reasoning", year=2024, arxiv="2310.04562", pdf_urls=["https://proceedings.iclr.cc/paper_files/paper/2024/file/85dd09d356ca561169b2c03e43cf305e-Paper-Conference.pdf"], relevance="Transferable relation reasoning across unseen KGs."),
    S("kgbet", "02_kgc_backbones_and_reasoning", "KG-BERT: BERT for Knowledge Graph Completion", year=2019, arxiv="1909.03193", relevance="Canonical text-based KGC baseline."),
    S("simkgc", "02_kgc_backbones_and_reasoning", "SimKGC: Simple Contrastive Knowledge Graph Completion with Pre-trained Language Models", year=2022, doi="10.18653/v1/2022.acl-long.295", arxiv="2203.02167", pdf_urls=["https://aclanthology.org/2022.acl-long.295.pdf"], relevance="Efficient text-based contrastive KGC baseline used by SATKGC."),
    S("satkgc", "02_kgc_backbones_and_reasoning", "Subgraph-Aware Training of Language Models for Knowledge Graph Completion Using Structure-Aware Contrastive Learning", year=2025, doi="10.1145/3696410.3714946", arxiv="2407.12703", relevance="Open-source recent text/structure KGC method already audited locally."),
    S("srpkgc", "02_kgc_backbones_and_reasoning", "Soft Reasoning Paths for Knowledge Graph Completion", year=2025, doi="10.24963/ijcai.2025/327", arxiv="2505.03285", pdf_urls=["https://www.ijcai.org/proceedings/2025/0327.pdf"], relevance="Recent soft-path KGC method already audited locally."),
    S("sat", "02_kgc_backbones_and_reasoning", "Enhancing Large Language Model for Knowledge Graph Completion via Structure-Aware Alignment-Tuning", year=2025, doi="10.18653/v1/2025.emnlp-main.1061", arxiv="2509.01166", pdf_urls=["https://aclanthology.org/2025.emnlp-main.1061.pdf"], relevance="Recent structure-aware LLM KGC method already audited locally."),
    S("flock", "02_kgc_backbones_and_reasoning", "Flock: A Knowledge Graph Foundation Model via Learning on Random Walks", year=2026, arxiv="2510.01510", relevance="Recent KG foundation-model comparison; high-resource rather than local baseline."),
    S("unihr", "02_kgc_backbones_and_reasoning", "UniHR: Hierarchical Representation Learning for Unified Knowledge Graph Link Prediction", year=2024, arxiv="2411.07019", relevance="Unified structured-fact link prediction; adjacent open-source method."),
    S("mhyper", "02_kgc_backbones_and_reasoning", "Collaboration of Fusion and Independence: Hypercomplex-driven Robust Multi-Modal Knowledge Graph Completion", year=2026, arxiv="2509.23714", pdf_urls=["https://aclanthology.org/2026.acl-long.1289.pdf"], relevance="Fusion-versus-independence mechanism comparison in multimodal KGC."),
    S("fustkgc", "02_kgc_backbones_and_reasoning", "FuST-KGC: Fusing Sub-graph Structures and Textual Semantics for Knowledge Graph Completion", year=2026, doi="10.1016/j.neucom.2026.132690", landing_url="https://doi.org/10.1016/j.neucom.2026.132690", relevance="Recent open-code structure/text fusion method audited locally; publisher PDF may require access."),
    S("faan", "02_kgc_backbones_and_reasoning", "Adaptive Attentional Network for Few-Shot Knowledge Graph Completion", year=2020, doi="10.18653/v1/2020.emnlp-main.131", pdf_urls=["https://aclanthology.org/2020.emnlp-main.131.pdf"], relevance="Low-resource KGC attention baseline."),
    S("sim_fewshot", "02_kgc_backbones_and_reasoning", "Semantic Interaction Matching Network for Few-Shot Knowledge Graph Completion", year=2024, doi="10.1145/3589557", relevance="Recent few-shot KGC method for low-resource comparisons."),

    # 03 — Negative transfer, source selection, and graph transfer.
    S("negative_transfer", "03_negative_transfer_and_source_selection", "Characterizing and Avoiding Negative Transfer", importance="core", year=2019, arxiv="1811.09751", pdf_urls=["https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Characterizing_and_Avoiding_Negative_Transfer_CVPR_2019_paper.pdf"], relevance="Foundational operational definition and mitigation of negative transfer."),
    S("negative_transfer_survey", "03_negative_transfer_and_source_selection", "A Survey on Negative Transfer", importance="core", year=2023, doi="10.1109/JAS.2022.106004", arxiv="2009.00909", relevance="Taxonomy of negative-transfer causes, detection, and mitigation."),
    S("transferability_survey", "03_negative_transfer_and_source_selection", "Transferability in Deep Learning: A Survey", year=2022, arxiv="2201.05867", relevance="Survey of source/model transferability estimation."),
    S("seval", "03_negative_transfer_and_source_selection", "Evaluating the Values of Sources in Transfer Learning", importance="core", year=2021, doi="10.18653/v1/2021.naacl-main.402", pdf_urls=["https://aclanthology.org/2021.naacl-main.402.pdf"], relevance="Closest source-valuation lineage; Shapley values over transfer sources."),
    S("share_sources", "03_negative_transfer_and_source_selection", "To Share or not to Share: Predicting Sets of Sources for Model Transfer Learning", importance="core", year=2021, doi="10.18653/v1/2021.emnlp-main.689", arxiv="2104.08078", pdf_urls=["https://aclanthology.org/2021.emnlp-main.689.pdf"], relevance="Predicts useful source sets and prevents negative transfer."),
    S("msda_moe", "03_negative_transfer_and_source_selection", "Multi-Source Domain Adaptation with Mixture of Experts", importance="core", year=2018, doi="10.18653/v1/D18-1498", pdf_urls=["https://aclanthology.org/D18-1498.pdf"], relevance="Instance-conditioned source expert weighting."),
    S("multisource_theory", "03_negative_transfer_and_source_selection", "Algorithms and Theory for Multiple-Source Adaptation", year=2018, pdf_urls=["https://proceedings.neurips.cc/paper/2018/file/2e2079d63348233d91cad1fa9b1361e9-Paper.pdf"], relevance="Formal multi-source weighting and adaptation guarantees."),
    S("winning_team", "03_negative_transfer_and_source_selection", "Building a Winning Team: Selecting Source Model Ensembles using a Submodular Transferability Estimation Approach", year=2023, pdf_urls=["https://openaccess.thecvf.com/content/ICCV2023/papers/B_Building_a_Winning_Team_Selecting_Source_Model_Ensembles_using_a_ICCV_2023_paper.pdf"], relevance="Subset-level source selection via transferability estimation."),
    S("data_based_transfer", "03_negative_transfer_and_source_selection", "A Data-Based Perspective on Transfer Learning", year=2023, arxiv="2207.05739", pdf_urls=["https://openaccess.thecvf.com/content/CVPR2023/papers/Jain_A_Data-Based_Perspective_on_Transfer_Learning_CVPR_2023_paper.pdf"], relevance="Empirical source data attribution and transfer behavior."),
    S("subgraph_pooling", "03_negative_transfer_and_source_selection", "Subgraph Pooling: Tackling Negative Transfer on Graphs", importance="core", year=2024, arxiv="2402.08907", pdf_urls=["https://www.ijcai.org/proceedings/2024/0570.pdf"], relevance="Closest graph-specific negative-transfer work."),
    S("noniid_graph_transfer", "03_negative_transfer_and_source_selection", "Non-IID Transfer Learning on Graphs", year=2023, doi="10.1609/aaai.v37i9.26231", relevance="Graph transfer under structural/distribution mismatch."),
    S("msds", "03_negative_transfer_and_source_selection", "MSDS: A Novel Framework for Multi-Source Data Selection Based Cross-Network Node Classification", year=2023, doi="10.1109/TKDE.2023.3277957", relevance="Multi-source graph-data selection under cross-network transfer."),
    S("best_metric", "03_negative_transfer_and_source_selection", "BeST: A Novel Source Selection Metric for Transfer Learning", year=2025, arxiv="2501.10933", relevance="Recent source-selection metric and novelty check."),
    S("automated_source_selection", "03_negative_transfer_and_source_selection", "On Automated Source Selection for Transfer Learning in Convolutional Neural Networks", year=2018, landing_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC6377173/", pdf_urls=["https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0201816&type=printable"], relevance="Automated source choice based on target/source behavior."),

    # 04 — Counterfactual utility, data value, abstention, and risk calibration.
    S("data_shapley", "04_counterfactual_utility_and_risk", "Data Shapley: Equitable Valuation of Data for Machine Learning", importance="core", year=2019, arxiv="1904.02868", pdf_urls=["https://proceedings.mlr.press/v97/ghorbani19c/ghorbani19c.pdf"], relevance="Canonical leave-subset-out utility allocation."),
    S("efficient_data_valuation", "04_counterfactual_utility_and_risk", "Towards Efficient Data Valuation Based on the Shapley Value", year=2019, arxiv="1902.10275", pdf_urls=["https://proceedings.mlr.press/v89/jia19a/jia19a.pdf"], relevance="Efficient approximations for intervention-based utility."),
    S("twod_shapley", "04_counterfactual_utility_and_risk", "2D-Shapley: A Framework for Fragmented Data Valuation", year=2023, arxiv="2306.10473", pdf_urls=["https://proceedings.mlr.press/v202/liu23s/liu23s.pdf"], relevance="Interaction-aware valuation across fragmented sources and samples."),
    S("data_oob", "04_counterfactual_utility_and_risk", "Data-OOB: Out-of-bag Estimate as a Simple and Efficient Data Value", year=2023, arxiv="2304.07718", pdf_urls=["https://proceedings.mlr.press/v202/kwon23e/kwon23e.pdf"], relevance="Cheap data-utility estimation alternative."),
    S("influence_functions", "04_counterfactual_utility_and_risk", "Understanding Black-box Predictions via Influence Functions", year=2017, arxiv="1703.04730", pdf_urls=["https://proceedings.mlr.press/v70/koh17a/koh17a.pdf"], relevance="Classical approximation to leave-one-out effects."),
    S("graph_shapley_utility", "04_counterfactual_utility_and_risk", "Shapley-Guided Utility Learning for Effective Graph Inference Data Valuation", importance="core", year=2025, pdf_urls=["https://proceedings.iclr.cc/paper_files/paper/2025/file/419b6c974712adb884bfbbeea8e94d1b-Paper-Conference.pdf"], relevance="Graph-specific learned utility prediction; important novelty threat."),
    S("dupre", "04_counterfactual_utility_and_risk", "DUPRE: Data Utility Prediction for Efficient Data Valuation", year=2025, pdf_urls=["https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p1557.pdf"], relevance="Learns to predict expensive utility labels; mechanism-adjacent novelty check."),
    S("learn_then_test", "04_counterfactual_utility_and_risk", "Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control", importance="core", year=2022, arxiv="2110.01052", relevance="Validation-driven selection under explicit risk constraints."),
    S("conformal_risk_control", "04_counterfactual_utility_and_risk", "Conformal Risk Control", year=2024, arxiv="2208.02814", pdf_urls=["https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf"], relevance="Finite-sample risk calibration framework relevant to certificate language."),
    S("selectivenet", "04_counterfactual_utility_and_risk", "SelectiveNet: A Deep Neural Network with an Integrated Reject Option", year=2019, arxiv="1901.09192", pdf_urls=["https://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf"], relevance="Learned abstention/selective prediction baseline."),
    S("selective_guarantees", "04_counterfactual_utility_and_risk", "Selective Classification for Deep Neural Networks", year=2017, arxiv="1705.08500", relevance="Risk-coverage control for abstaining predictors."),

    # 05 — Evaluation protocols, surveys, and reproducibility tooling.
    S("kge_survey", "05_evaluation_surveys_and_systems", "Knowledge Graph Embedding: A Survey of Approaches and Applications", year=2017, doi="10.1109/TKDE.2017.2754499", arxiv="1709.07604", relevance="Canonical KGE taxonomy and terminology."),
    S("kgc_reevaluation", "05_evaluation_surveys_and_systems", "A Re-evaluation of Knowledge Graph Completion Methods", importance="core", year=2020, doi="10.18653/v1/2020.acl-main.489", arxiv="1911.03903", pdf_urls=["https://aclanthology.org/2020.acl-main.489.pdf"], relevance="Hyperparameter and training-protocol fairness for KGC baselines."),
    S("kgc_eval_protocol", "05_evaluation_surveys_and_systems", "Revisiting the Evaluation Protocol of Knowledge Graph Completion Methods for Link Prediction", importance="core", year=2021, doi="10.1145/3442381.3449856", relevance="Filtered ranking protocol pitfalls and realistic evaluation."),
    S("kgc_eval_ir", "05_evaluation_surveys_and_systems", "Re-thinking Knowledge Graph Completion Evaluation from an Information Retrieval Perspective", importance="core", year=2022, doi="10.1145/3477495.3532052", arxiv="2205.04105", relevance="Shows how query sparsity and IR-style evaluation change KGC conclusions."),
    S("rank_metrics_framework", "05_evaluation_surveys_and_systems", "A Unified Framework for Rank-based Evaluation Metrics for Link Prediction in Knowledge Graphs", importance="core", year=2022, arxiv="2203.07544", pdf_urls=["https://graph-learning-benchmarks.github.io/assets/papers/glb2022/A_Unified_Framework_for_Rank_based_Evaluation_Metrics_for_Link_Prediction_in_Knowledge_Graphs.pdf"], relevance="Formalizes rank metrics, expectations, variance, and adjusted alternatives."),
    S("kgc_model_calibration", "05_evaluation_surveys_and_systems", "Using Model Calibration to Evaluate Link Prediction in Knowledge Graphs", year=2024, doi="10.1145/3589334.3645506", pdf_urls=["https://openreview.net/pdf?id=8f8GrRqb2l"], landing_url="https://openreview.net/forum?id=8f8GrRqb2l", relevance="Calibration-oriented KGC evaluation adjacent to the utility-threshold audit."),
    S("open_world_eval", "05_evaluation_surveys_and_systems", "Rethinking Knowledge Graph Evaluation Under the Open-World Assumption", year=2022, pdf_urls=["https://proceedings.neurips.cc/paper_files/paper/2022/file/378226e5df7eded3e401de5c9493143c-Paper-Conference.pdf"], relevance="Evaluation when unobserved triples may still be true."),
    S("birds_eye_kge", "05_evaluation_surveys_and_systems", "A Bird's Eye View on Knowledge Graph Embeddings, Software Libraries, Applications and Challenges", year=2022, arxiv="2205.09088", relevance="Modern survey with software and benchmarking perspective."),
    S("unseen_elements_survey", "05_evaluation_surveys_and_systems", "Generalizing to Unseen Elements: A Survey on Knowledge Extrapolation for Knowledge Graphs", year=2023, doi="10.24963/ijcai.2023/737", arxiv="2302.01859", pdf_urls=["https://www.ijcai.org/proceedings/2023/0737.pdf"], relevance="Inductive/extrapolative KGC taxonomy."),
    S("kge_review_2023", "05_evaluation_surveys_and_systems", "Knowledge Graph Embedding: An Overview", year=2023, arxiv="2309.12501", relevance="Recent broad KGE/KGC overview."),
    S("fb15k237", "05_evaluation_surveys_and_systems", "Representing Text for Joint Embedding of Text and Knowledge Bases", year=2015, arxiv="1509.01669", pdf_urls=["https://aclanthology.org/D15-1174.pdf"], relevance="Introduces the leakage-reduced FB15k-237 benchmark lineage."),
    S("pykeen", "05_evaluation_surveys_and_systems", "PyKEEN 1.0: A Python Library for Training and Evaluating Knowledge Graph Embeddings", year=2021, arxiv="2007.14175", relevance="Reference implementation and reproducible KGE evaluation tooling."),
    S("dglke", "05_evaluation_surveys_and_systems", "DGL-KE: Training Knowledge Graph Embeddings at Scale", year=2020, arxiv="2004.08532", relevance="Scalable KGE systems reference for the resource audit."),
]


def request_bytes(url: str, *, accept: str | None = None, timeout: int = 35) -> tuple[bytes, str, str]:
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(), resp.headers.get("Content-Type", ""), resp.geturl()


def request_json(url: str, timeout: int = 35) -> dict[str, Any] | None:
    try:
        raw, _, _ = request_bytes(url, accept="application/json", timeout=timeout)
        return json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return None


def norm_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def title_score(a: str, b: str) -> float:
    na, nb = norm_text(a), norm_text(b)
    if not na or not nb:
        return 0.0
    seq = SequenceMatcher(None, na, nb).ratio()
    sa, sb = set(na.split()), set(nb.split())
    jac = len(sa & sb) / max(1, len(sa | sb))
    return 0.55 * seq + 0.45 * jac


def clean_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = doi.strip()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.I)
    return doi.lower()


def openalex_record(seed: Seed) -> dict[str, Any] | None:
    base = "https://api.openalex.org/works"
    data: dict[str, Any] | None = None
    if seed.doi:
        url = f"{base}/https://doi.org/{urllib.parse.quote(clean_doi(seed.doi) or '', safe='/()')}?mailto={urllib.parse.quote(CONTACT)}"
        data = request_json(url)
        if data and data.get("id"):
            return data
    params = urllib.parse.urlencode({"search": seed.title, "per-page": 5, "mailto": CONTACT})
    result = request_json(f"{base}?{params}")
    rows = (result or {}).get("results", [])
    if not rows:
        return None
    ranked = sorted(rows, key=lambda row: title_score(seed.title, row.get("title") or ""), reverse=True)
    if title_score(seed.title, ranked[0].get("title") or "") < 0.58:
        return None
    return ranked[0]


def crossref_record(doi: str | None) -> dict[str, Any] | None:
    doi = clean_doi(doi)
    if not doi:
        return None
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/()")
    data = request_json(url)
    return (data or {}).get("message")


def unpaywall_record(doi: str | None) -> dict[str, Any] | None:
    doi = clean_doi(doi)
    if not doi:
        return None
    url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi, safe='/()')}?email={urllib.parse.quote(CONTACT)}"
    return request_json(url)


def arxiv_metadata(arxiv_id: str | None) -> dict[str, Any] | None:
    if not arxiv_id:
        return None
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"id_list": arxiv_id})
    try:
        raw, _, _ = request_bytes(url, timeout=30)
        root = ET.fromstring(raw)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entry = root.find("a:entry", ns)
        if entry is None:
            return None
        title = " ".join((entry.findtext("a:title", default="", namespaces=ns)).split())
        authors = [a.findtext("a:name", default="", namespaces=ns) for a in entry.findall("a:author", ns)]
        published = entry.findtext("a:published", default="", namespaces=ns)
        return {"title": title, "authors": authors, "published": published}
    except Exception:
        return None


def authors_from_openalex(row: dict[str, Any] | None) -> list[str]:
    result: list[str] = []
    for authorship in (row or {}).get("authorships", []):
        name = ((authorship or {}).get("author") or {}).get("display_name")
        if name:
            result.append(name)
    return result


def authors_from_crossref(row: dict[str, Any] | None) -> list[str]:
    result: list[str] = []
    for author in (row or {}).get("author", []):
        name = " ".join(x for x in [author.get("given", ""), author.get("family", "")] if x).strip()
        if name:
            result.append(name)
    return result


def collect_pdf_candidates(seed: Seed, oa: dict[str, Any] | None, upw: dict[str, Any] | None, cr: dict[str, Any] | None) -> list[str]:
    urls = list(seed.pdf_urls)
    if seed.arxiv:
        aid = re.sub(r"v\d+$", "", seed.arxiv)
        urls += [f"https://arxiv.org/pdf/{aid}.pdf", f"https://export.arxiv.org/pdf/{aid}.pdf"]
    if seed.doi and "10.18653/v1/" in seed.doi.lower():
        anthology_id = seed.doi.split("/", 1)[1]
        urls.append(f"https://aclanthology.org/{anthology_id}.pdf")
    for location in [
        (oa or {}).get("best_oa_location"),
        (oa or {}).get("primary_location"),
        *((oa or {}).get("locations") or []),
    ]:
        if isinstance(location, dict):
            if location.get("pdf_url"):
                urls.append(location["pdf_url"])
            landing = location.get("landing_page_url") or ""
            m = re.search(r"arxiv\.org/(?:abs|html)/(\d{4}\.\d{4,5})(?:v\d+)?", landing)
            if m:
                urls.append(f"https://arxiv.org/pdf/{m.group(1)}.pdf")
    for location in [(upw or {}).get("best_oa_location"), *((upw or {}).get("oa_locations") or [])]:
        if isinstance(location, dict):
            if location.get("url_for_pdf"):
                urls.append(location["url_for_pdf"])
            elif location.get("url") and str(location.get("url")).lower().endswith(".pdf"):
                urls.append(location["url"])
    for link in (cr or {}).get("link", []):
        if not isinstance(link, dict) or not link.get("URL"):
            continue
        content_type = str(link.get("content-type") or "").lower()
        if "pdf" in content_type or str(link["URL"]).lower().split("?", 1)[0].endswith(".pdf") or "/doi/pdf/" in str(link["URL"]).lower():
            urls.append(link["URL"])
    # Common conversions for explicitly supplied landing pages.
    for candidate in [seed.landing_url or ""]:
        m = re.search(r"arxiv\.org/(?:abs|html)/(\d{4}\.\d{4,5})(?:v\d+)?", candidate)
        if m:
            urls.append(f"https://arxiv.org/pdf/{m.group(1)}.pdf")
    seen: set[str] = set()
    out: list[str] = []
    for url in urls:
        if not url:
            continue
        url = html.unescape(url).replace("http://arxiv.org/", "https://arxiv.org/")
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def safe_filename(seed: Seed, title: str, year: int | None) -> str:
    stem = norm_text(title).replace(" ", "_")[:95].strip("_") or seed.key
    prefix = str(year or seed.year or "unknown")
    return f"{prefix}_{seed.key}_{stem}.pdf"


def is_pdf(data: bytes) -> bool:
    return len(data) >= 10_000 and b"%PDF-" in data[:1024]


def download_pdf(url: str, destination: Path) -> tuple[bool, str, str | None]:
    try:
        data, content_type, final_url = request_bytes(url, accept="application/pdf,*/*;q=0.8", timeout=55)
        if not is_pdf(data):
            return False, f"not a PDF ({content_type or 'unknown content type'}, {len(data)} bytes)", final_url
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(destination.suffix + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, destination)
        return True, f"downloaded {len(data)} bytes", final_url
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}", getattr(exc, "url", url)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", None


def bib_escape(value: str) -> str:
    return value.replace("\\", "\\textbackslash{}") .replace("{", "\\{").replace("}", "\\}").replace("&", "\\&")


def make_bibtex(row: dict[str, Any]) -> str:
    authors = row.get("authors") or []
    fields = {
        "title": row.get("title"),
        "author": " and ".join(authors) if authors else None,
        "year": str(row.get("year")) if row.get("year") else None,
        "doi": row.get("doi"),
        "eprint": row.get("arxiv"),
        "url": row.get("landing_url") or row.get("source_url"),
        "note": row.get("relevance"),
    }
    if row.get("arxiv"):
        fields["archivePrefix"] = "arXiv"
        fields["primaryClass"] = row.get("primary_class")
    body = []
    for key, value in fields.items():
        if value:
            body.append(f"  {key} = {{{bib_escape(str(value))}}}")
    return "@article{" + row["key"] + ",\n" + ",\n".join(body) + "\n}"


def resolve_and_download(seed: Seed) -> dict[str, Any]:
    cached = PREVIOUS.get(seed.key)
    if cached and cached.get("status") == "downloaded" and cached.get("local_path"):
        cached_path = ROOT / cached["local_path"]
        try:
            if cached_path.exists() and is_pdf(cached_path.read_bytes()):
                print(f"[cached] {seed.key}: {cached_path.stat().st_size:,} bytes", flush=True)
                return cached
        except Exception:
            pass
    print(f"[resolve] {seed.key}: {seed.title}", flush=True)
    oa = openalex_record(seed)
    doi = clean_doi(seed.doi or ((oa or {}).get("doi") or ""))
    cr = crossref_record(doi)
    upw = unpaywall_record(doi)
    ax = arxiv_metadata(seed.arxiv)

    title = seed.title
    if oa and title_score(seed.title, oa.get("title") or "") >= 0.70:
        title = oa.get("title") or title
    authors = authors_from_openalex(oa) or authors_from_crossref(cr) or (ax or {}).get("authors") or []
    year = seed.year or (oa or {}).get("publication_year")
    if not year and cr:
        parts = (((cr.get("published-print") or cr.get("published-online") or {}).get("date-parts") or [[None]])[0])
        year = parts[0] if parts else None
    if not year and (ax or {}).get("published"):
        try:
            year = int(str(ax["published"])[:4])
        except Exception:
            pass

    landing = seed.landing_url
    if not landing:
        landing = ((oa or {}).get("primary_location") or {}).get("landing_page_url")
    if not landing and doi:
        landing = f"https://doi.org/{doi}"
    if not landing and seed.arxiv:
        landing = f"https://arxiv.org/abs/{seed.arxiv}"

    destination = PAPERS / seed.theme / safe_filename(seed, title, year)
    attempts: list[dict[str, str]] = []
    status = "missing"
    source_url: str | None = None
    error = "No public PDF candidate found"

    if seed.local_source:
        source_path = ROOT / seed.local_source
        if source_path.exists() and is_pdf(source_path.read_bytes()):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
            status = "downloaded"
            source_url = str(source_path)
            error = ""
            attempts.append({"url": str(source_path), "result": "copied verified local PDF"})
        else:
            attempts.append({"url": str(source_path), "result": "local PDF missing or invalid"})

    if status != "downloaded":
        for url in collect_pdf_candidates(seed, oa, upw, cr):
            ok, message, final_url = download_pdf(url, destination)
            attempts.append({"url": url, "result": message})
            if ok:
                status = "downloaded"
                source_url = final_url or url
                error = ""
                break
            error = message
            time.sleep(0.12)

    size = destination.stat().st_size if destination.exists() else 0
    sha256 = hashlib.sha256(destination.read_bytes()).hexdigest() if destination.exists() else ""
    oa_status = ((oa or {}).get("open_access") or {}).get("oa_status") or (upw or {}).get("oa_status") or "unknown"
    row = {
        "key": seed.key,
        "theme": seed.theme,
        "importance": seed.importance,
        "title": title,
        "authors": authors,
        "year": year,
        "doi": doi,
        "arxiv": seed.arxiv,
        "landing_url": landing,
        "status": status,
        "local_path": str(destination.relative_to(ROOT)) if destination.exists() else "",
        "size_bytes": size,
        "sha256": sha256,
        "source_url": source_url,
        "oa_status": oa_status,
        "relevance": seed.relevance,
        "error": error,
        "attempts": attempts,
        "openalex_id": (oa or {}).get("id"),
    }
    print(f"[{status}] {seed.key}: {size:,} bytes" if status == "downloaded" else f"[missing] {seed.key}: {error}", flush=True)
    return row


def write_outputs(rows: list[dict[str, Any]], elapsed: float) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["theme"], 0 if r["importance"] == "core" else 1, -(r.get("year") or 0), r["title"]))
    (OUT / "literature_manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    columns = ["key", "theme", "importance", "title", "authors", "year", "doi", "arxiv", "status", "local_path", "size_bytes", "sha256", "oa_status", "landing_url", "source_url", "relevance", "error"]
    with (OUT / "literature_catalog.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key, "") for key in columns}
            out["authors"] = "; ".join(row.get("authors") or [])
            writer.writerow(out)

    bib = "\n\n".join(make_bibtex(row) for row in rows) + "\n"
    (OUT / "references.bib").write_text(bib, encoding="utf-8")

    downloaded = [row for row in rows if row["status"] == "downloaded"]
    missing = [row for row in rows if row["status"] != "downloaded"]
    core_missing = [row for row in missing if row["importance"] == "core"]
    checksums = "\n".join(f"{row['sha256']}  {row['local_path'].replace(os.sep, '/')}" for row in downloaded) + "\n"
    (OUT / "checksums.sha256").write_text(checksums, encoding="utf-8")

    missing_lines = [
        "# 需要手动下载的文献",
        "",
        "下列条目没有找到可直接、合法访问的公开 PDF。优先打开 DOI/论文主页；若你通过学校或机构有访问权限，可下载后按建议文件名放入对应主题目录。",
        "",
        f"共 {len(missing)} 篇，其中核心文献 {len(core_missing)} 篇。",
        "",
    ]
    for row in missing:
        tags = "核心" if row["importance"] == "core" else "补充"
        missing_lines += [
            f"## {row.get('year') or '年份未知'} · {row['title']}",
            "",
            f"- 级别：{tags}",
            f"- 主题：`{row['theme']}`",
            f"- DOI：{('https://doi.org/' + row['doi']) if row.get('doi') else '未找到'}",
            f"- arXiv：{('https://arxiv.org/abs/' + row['arxiv']) if row.get('arxiv') else '未找到'}",
            f"- 论文页：{row.get('landing_url') or '未找到'}",
            f"- 建议目录：`papers/{row['theme']}/`",
            f"- 原因：{row.get('error') or '下载失败'}",
            "",
        ]
    (OUT / "manual_download_links.md").write_text("\n".join(missing_lines), encoding="utf-8")

    themes: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        themes.setdefault(row["theme"], []).append(row)
    readme = [
        "# QURA-Cert 文献包",
        "",
        "本目录围绕多领域知识图谱补全、负迁移与源选择、查询级反事实效用、KGC 基线与评测协议整理。PDF 仅来自公开作者稿、开放获取出版页、arXiv、ACL Anthology、PMLR、OpenReview、IJCAI、NeurIPS/CVF 等合法公开来源。",
        "",
        "## 汇总",
        "",
        f"- 候选文献：{len(rows)} 篇",
        f"- 已获得并验证 PDF：{len(downloaded)} 篇",
        f"- 需要手动下载：{len(missing)} 篇",
        f"- 核心文献缺失：{len(core_missing)} 篇",
        f"- 运行耗时：{elapsed:.1f} 秒",
        "",
        "## 文件",
        "",
        "- `references.bib`：便携版 BibTeX，可直接用于 LaTeX/JabRef。",
        "- `references_zotero.bib`：含本机 PDF 绝对路径，导入 Zotero 时可一并关联附件。",
        "- `references.ris`：适合 EndNote、NoteExpress、Zotero 等引用管理器。",
        "- `literature_catalog.csv`：选题、重要性、DOI/arXiv、下载状态、相关性说明。",
        "- `manual_download_links.md`：未能自动取得 PDF 的 DOI 和论文页。",
        "- `literature_manifest.json`：完整机器可读记录，含每个下载尝试。",
        "- `checksums.sha256`：PDF 完整性校验。",
        "- `citation_metadata_report.md`：作者和 venue 完整性检查结果。",
        "",
        "## 主题统计",
        "",
        "| 主题 | 总数 | 已下载 | 缺失 |",
        "|---|---:|---:|---:|",
    ]
    for theme, items in sorted(themes.items()):
        n_ok = sum(x["status"] == "downloaded" for x in items)
        readme.append(f"| `{theme}` | {len(items)} | {n_ok} | {len(items)-n_ok} |")
    readme += [
        "",
        "## 使用建议",
        "",
        "先读 `01_core_multidomain_kgc` 建立 Related Work 主线，再读 `03_negative_transfer_and_source_selection` 与 `04_counterfactual_utility_and_risk` 做新颖性边界；实验实现与评价协议集中在 `02_kgc_backbones_and_reasoning` 和 `05_evaluation_surveys_and_systems`。",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(readme), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    print(f"Preparing {len(SEEDS)} curated references under {OUT}", flush=True)
    rows: list[dict[str, Any]] = []
    # Modest concurrency avoids hammering scholarly APIs and publisher sites.
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(resolve_and_download, seed): seed for seed in SEEDS}
        for future in as_completed(futures):
            seed = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                print(f"[fatal-item] {seed.key}: {type(exc).__name__}: {exc}", flush=True)
                rows.append({
                    "key": seed.key, "theme": seed.theme, "importance": seed.importance,
                    "title": seed.title, "authors": [], "year": seed.year,
                    "doi": clean_doi(seed.doi), "arxiv": seed.arxiv,
                    "landing_url": seed.landing_url or (f"https://doi.org/{seed.doi}" if seed.doi else (f"https://arxiv.org/abs/{seed.arxiv}" if seed.arxiv else None)),
                    "status": "missing", "local_path": "", "size_bytes": 0,
                    "sha256": "", "source_url": None, "oa_status": "unknown",
                    "relevance": seed.relevance, "error": f"pipeline error: {exc}",
                    "attempts": [], "openalex_id": None,
                })
    elapsed = time.monotonic() - start
    write_outputs(rows, elapsed)
    ok = sum(row["status"] == "downloaded" for row in rows)
    print(f"DONE: {ok}/{len(rows)} PDFs available; {len(rows)-ok} require manual retrieval; {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
